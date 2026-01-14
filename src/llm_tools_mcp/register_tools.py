import asyncio
import atexit
import threading
import llm
import mcp
from pathlib import Path
from typing import Optional, Any


from llm_tools_mcp.defaults import DEFAULT_MCP_JSON_PATH
from llm_tools_mcp.mcp_config import McpConfig
from llm_tools_mcp.mcp_client import McpClient
from llm_tools_mcp.cache import load_cached_tools, save_tools_cache


# =============================================================================
# Schema Sanitization for Gemini Compatibility
# =============================================================================
# Gemini's function calling API doesn't support all JSON Schema keywords.
# We strip unsupported keywords to prevent "Unknown name" errors.

UNSUPPORTED_SCHEMA_KEYWORDS = {
    # Numeric constraints not supported by Gemini
    "exclusiveMinimum",
    "exclusiveMaximum",
    # Schema composition keywords (partial support)
    "$ref",
    "$defs",
    "definitions",
    # Format/content keywords
    "contentMediaType",
    "contentEncoding",
    # Deprecated keywords
    "dependencies",
}


def _sanitize_schema(schema: dict | Any) -> dict | Any:
    """Recursively remove unsupported JSON Schema keywords for Gemini compatibility."""
    if not isinstance(schema, dict):
        return schema

    sanitized = {}
    for key, value in schema.items():
        if key in UNSUPPORTED_SCHEMA_KEYWORDS:
            continue  # Skip unsupported keywords

        if isinstance(value, dict):
            sanitized[key] = _sanitize_schema(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_schema(item) if isinstance(item, (dict, list)) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


# =============================================================================
# Thread-Local Event Loop Management
# =============================================================================
# MCP sessions and anyio cancel scopes are bound to event loops. When tool calls
# come from different threads (e.g., daemon's thread pool executor), each thread
# needs its own event loop to avoid cross-thread issues.
#
# Strategy:
# - _run_async_init(): Used during MCP initialization (tool discovery)
#   Creates a temporary loop that's discarded after init
# - _run_async(): Used for tool calls, uses thread-local loops
#   Each thread gets its own persistent loop for session reuse within that thread
#
# Note: This means sessions are per-thread. In the Terminator TUI case (single
# thread), sessions are fully reused. In the daemon case (thread pool), each
# worker thread maintains its own sessions.

_loop_local = threading.local()  # Thread-local storage for event loops
_mcp_client: Optional[McpClient] = None
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Get or create a thread-local event loop."""
    global _cleanup_registered
    if not hasattr(_loop_local, 'loop') or _loop_local.loop is None or _loop_local.loop.is_closed():
        _loop_local.loop = asyncio.new_event_loop()
        # Register cleanup only once (first thread to create a loop)
        with _cleanup_lock:
            if not _cleanup_registered:
                atexit.register(_cleanup_loop)
                _cleanup_registered = True
    return _loop_local.loop


def _run_async_init(coro):
    """Run coroutine during initialization using a temporary loop.

    Used for tool discovery - creates and discards a temporary event loop
    to avoid polluting thread-local loops with init thread ownership.
    """
    temp_loop = asyncio.new_event_loop()
    try:
        return temp_loop.run_until_complete(coro)
    finally:
        temp_loop.close()


def _run_async(coro):
    """Run coroutine on the thread-local loop.

    Each thread gets its own event loop, ensuring anyio cancel scopes
    work correctly. Sessions are reused within each thread.
    """
    loop = _get_thread_loop()
    return loop.run_until_complete(coro)


def _cleanup_loop():
    """Cleanup registered with atexit - closes MCP client sessions.

    With thread-local loops, we can't enumerate all loops. Instead, we close
    the MCP client's sessions using a temporary loop. The thread-local loops
    are cleaned up when their threads exit.
    """
    global _mcp_client
    if _mcp_client:
        try:
            # Use a temporary loop for cleanup - thread-local loops may not exist
            # in the main thread at exit time
            temp_loop = asyncio.new_event_loop()
            try:
                temp_loop.run_until_complete(_mcp_client.close_all())
            finally:
                temp_loop.close()
        except Exception:
            pass  # Best effort cleanup


# =============================================================================
# Tool Creation
# =============================================================================

def _create_tool_for_mcp(
    server_name: str, mcp_client: McpClient, mcp_config: McpConfig, mcp_tool: mcp.Tool
) -> llm.Tool:
    """Create an llm.Tool that calls the MCP server via persistent session."""
    def impl(**kwargs):
        return _run_async(mcp_client.call_tool(server_name, mcp_tool.name, **kwargs))

    enriched_description = mcp_tool.description or ""
    enriched_description += f"\n[from MCP server: {server_name}]"

    tool = llm.Tool(
        name=mcp_tool.name,
        description=enriched_description,
        input_schema=_sanitize_schema(mcp_tool.inputSchema),
        plugin="llm-tools-mcp",
        implementation=impl,
    )
    # Store server name and optional status for filtering
    tool.server_name = server_name
    tool.mcp_optional = mcp_config.is_optional(server_name)
    return tool


def _get_tools_for_llm(mcp_client: McpClient, mcp_config: McpConfig) -> tuple[list[llm.Tool], bool]:
    """Fetch tools from all MCP servers and convert to llm.Tool objects.

    Called during initialization - uses temporary loop to avoid polluting
    the persistent loop with background thread ownership.

    Returns:
        Tuple of (tools_list, had_errors) where had_errors is True if any server failed.
    """
    tools, had_errors = _run_async_init(mcp_client.get_all_tools())
    mapped_tools: list[llm.Tool] = []
    for server_name, server_tools in tools.items():
        for tool in server_tools:
            if not mcp_config.should_include_tool(server_name, tool.name):
                continue
            mapped_tools.append(_create_tool_for_mcp(server_name, mcp_client, mcp_config, tool))
    return mapped_tools, had_errors


# =============================================================================
# Tool Caching Helpers
# =============================================================================

def _serialize_tools(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    """Serialize tool schemas to JSON-compatible format for caching."""
    return [
        {
            "server": getattr(tool, 'server_name', 'unknown'),
            "name": tool.name,
            "description": tool.description or "",
            "schema": tool.input_schema,
        }
        for tool in tools
    ]


def _rebuild_tools_from_cache(
    cached: list[dict[str, Any]],
    mcp_client: McpClient,
    mcp_config: McpConfig
) -> list[llm.Tool]:
    """Recreate llm.Tool objects from cached schemas.

    Note: The implementation still makes actual MCP calls - only schemas are cached.
    """
    tools = []
    for item in cached:
        server_name = item["server"]
        tool_name = item["name"]

        # Respect current include/exclude filters
        if not mcp_config.should_include_tool(server_name, tool_name):
            continue

        # Create implementation that calls the actual MCP server
        # Use default args to capture current values in closure
        def impl(server=server_name, name=tool_name, **kwargs):
            return _run_async(mcp_client.call_tool(server, name, **kwargs))

        description = item["description"]
        if not description.endswith(f"\n[from MCP server: {server_name}]"):
            description += f"\n[from MCP server: {server_name}]"

        tool = llm.Tool(
            name=tool_name,
            description=description,
            input_schema=_sanitize_schema(item["schema"]),
            plugin="llm-tools-mcp",
            implementation=impl,
        )
        tool.server_name = server_name
        tool.mcp_optional = mcp_config.is_optional(server_name)
        tools.append(tool)

    return tools


# =============================================================================
# Toolbox Registration
# =============================================================================

class MCP(llm.Toolbox):
    """MCP Toolbox with session persistence and tool caching.

    Features:
    - Session persistence: Sessions reused across tool calls (70-80% faster)
    - Tool caching: Tool schemas cached to disk (instant subsequent starts)
    - Parallel init: Multiple servers connected concurrently
    """

    def __init__(self, config_path: str = DEFAULT_MCP_JSON_PATH):
        global _mcp_client

        # Read config file content for cache key
        config_file = Path(config_path).expanduser()
        config_content = config_file.read_text()

        mcp_config = McpConfig.for_json_content(config_content)
        mcp_client = McpClient(mcp_config)

        # Store globally for cleanup
        _mcp_client = mcp_client

        # Try cache first - schemas only, not connections
        cached = load_cached_tools(config_content)
        if cached:
            # Recreate llm.Tool objects from cached schemas
            # Implementation still calls mcp_client.call_tool() on invocation
            computed_tools = _rebuild_tools_from_cache(cached, mcp_client, mcp_config)
        else:
            # Fetch schemas from servers (connects to each in parallel)
            computed_tools, had_errors = _get_tools_for_llm(mcp_client, mcp_config)
            # Only cache if no errors - ensures failed servers are retried next time
            if not had_errors:
                save_tools_cache(config_content, _serialize_tools(computed_tools))

        for tool in computed_tools:
            self.add_tool(tool, pass_self=True)


@llm.hookimpl
def register_tools(register):
    register(MCP)

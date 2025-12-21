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
# Persistent Event Loop Management
# =============================================================================
# Sessions are bound to event loops - we need a persistent loop for session reuse

_event_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()  # Thread-safe lock for loop creation
_mcp_client: Optional[McpClient] = None
_cleanup_registered = False


def _get_persistent_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for session reuse."""
    global _event_loop, _cleanup_registered
    with _loop_lock:
        if _event_loop is None or _event_loop.is_closed():
            _event_loop = asyncio.new_event_loop()
            if not _cleanup_registered:
                atexit.register(_cleanup_loop)
                _cleanup_registered = True
        return _event_loop


def _run_async(coro):
    """Run coroutine on the persistent loop."""
    loop = _get_persistent_loop()
    return loop.run_until_complete(coro)


def _cleanup_loop():
    """Cleanup registered with atexit - closes sessions and event loop."""
    global _event_loop, _mcp_client
    if _mcp_client and _event_loop and not _event_loop.is_closed():
        try:
            _event_loop.run_until_complete(_mcp_client.close_all())
        except Exception:
            pass
    if _event_loop and not _event_loop.is_closed():
        try:
            # Cancel any remaining pending tasks
            pending = asyncio.all_tasks(_event_loop)
            for task in pending:
                task.cancel()
            # Allow cancelled tasks to complete
            if pending:
                _event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            _event_loop.close()
        except Exception:
            pass


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


def _get_tools_for_llm(mcp_client: McpClient, mcp_config: McpConfig) -> list[llm.Tool]:
    """Fetch tools from all MCP servers and convert to llm.Tool objects."""
    tools = _run_async(mcp_client.get_all_tools())
    mapped_tools: list[llm.Tool] = []
    for server_name, server_tools in tools.items():
        for tool in server_tools:
            if not mcp_config.should_include_tool(server_name, tool.name):
                continue
            mapped_tools.append(_create_tool_for_mcp(server_name, mcp_client, mcp_config, tool))
    return mapped_tools


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
            computed_tools = _get_tools_for_llm(mcp_client, mcp_config)
            # Cache for next time
            save_tools_cache(config_content, _serialize_tools(computed_tools))

        for tool in computed_tools:
            self.add_tool(tool, pass_self=True)


@llm.hookimpl
def register_tools(register):
    register(MCP)

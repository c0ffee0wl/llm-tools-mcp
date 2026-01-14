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
# Persistent Event Loop for MCP Operations
# =============================================================================
# MCP sessions use anyio cancel scopes which are bound to TASK CONTEXTS, not
# just event loops. Each run_until_complete() call creates a fresh task context.
# When we reuse a session across multiple run_until_complete() calls, the cancel
# scopes from the first call become invalid in subsequent calls.
#
# Solution: Keep a single event loop running continuously in a background thread
# using run_forever(). All MCP operations are submitted via run_coroutine_threadsafe(),
# which schedules them on the persistent loop without creating new task contexts.
#
# This ensures:
# - Sessions are created and used in compatible task contexts
# - Cancel scopes remain valid across multiple tool calls
# - Thread-safe access from any calling thread (main, executor, etc.)


class PersistentLoopRunner:
    """Runs an event loop in a background thread, keeping task contexts alive.

    This solves the anyio cancel scope issue: since the loop runs continuously
    (via run_forever()), all coroutines execute in compatible task contexts.
    Sessions can be reused across multiple calls without cancel scope errors.
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._lock = threading.Lock()

    def _ensure_started(self):
        """Start the background thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._loop = asyncio.new_event_loop()
            self._started.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="mcp-event-loop"
            )
            self._thread.start()
            self._started.wait()  # Wait for loop to start

    def _run_loop(self):
        """Background thread: set up and run the event loop forever."""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro):
        """Submit a coroutine and wait for result.

        Can be called from any thread. The coroutine runs on the persistent
        loop's thread, ensuring cancel scopes remain valid.
        """
        self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self):
        """Stop the event loop and join the thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


_loop_runner: Optional[PersistentLoopRunner] = None
_mcp_client: Optional[McpClient] = None
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _get_loop_runner() -> PersistentLoopRunner:
    """Get or create the global persistent loop runner."""
    global _loop_runner, _cleanup_registered
    if _loop_runner is None:
        with _cleanup_lock:
            if _loop_runner is None:
                _loop_runner = PersistentLoopRunner()
                if not _cleanup_registered:
                    atexit.register(_cleanup)
                    _cleanup_registered = True
    return _loop_runner


def _run_async_init(coro):
    """Run coroutine during initialization using a temporary loop.

    Used for tool discovery - creates and discards a temporary event loop.
    Tool discovery uses ephemeral sessions, so no persistent state is created.
    """
    temp_loop = asyncio.new_event_loop()
    try:
        return temp_loop.run_until_complete(coro)
    finally:
        temp_loop.close()


def _run_async(coro):
    """Run coroutine on the persistent loop.

    All MCP tool calls go through here. The persistent loop keeps running
    (via run_forever()), so task contexts and cancel scopes remain valid
    across multiple calls. This is the key to session reuse.
    """
    return _get_loop_runner().run(coro)


def _cleanup():
    """Cleanup registered with atexit - closes MCP sessions and stops loop."""
    global _mcp_client, _loop_runner

    if _mcp_client is not None and _loop_runner is not None:
        try:
            # Close sessions on the persistent loop (where they live)
            _loop_runner.run(_mcp_client.close_all())
        except Exception:
            pass  # Best effort cleanup

    if _loop_runner is not None:
        _loop_runner.stop()


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

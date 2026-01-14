from llm_tools_mcp.mcp_config import McpConfig, SseServerConfig, StdioServerConfig
from llm_tools_mcp.mcp_config import HttpServerConfig
import asyncio
import sys
import threading

from mcp import (
    ClientSession,
    ListToolsResult,
    StdioServerParameters,
    Tool,
    stdio_client,
)
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


import datetime
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, TextIO, Optional


class McpClient:
    """MCP client with session persistence for improved performance.

    Sessions are cached per-server and reused across tool calls, eliminating
    the 300-500ms overhead of creating a new session for each call.

    All operations run on a single persistent event loop (managed by
    register_tools.PersistentLoopRunner), which keeps task contexts alive
    and ensures anyio cancel scopes remain valid across calls.
    """

    def __init__(self, config: McpConfig):
        self.config = config
        # Session storage - all accessed from the persistent loop's thread
        self._sessions: dict[str, ClientSession] = {}
        self._contexts: dict[str, Any] = {}
        self._http_sessions: dict[str, Any] = {}
        self._read_streams: dict[str, Any] = {}
        self._init_lock = threading.Lock()  # Thread-safe initialization

    async def _get_or_create_session(self, name: str) -> Optional[ClientSession]:
        """Get cached session or create new one with proper lifecycle."""
        if name in self._sessions:
            return self._sessions[name]

        # Double-checked locking pattern for thread safety
        with self._init_lock:
            if name in self._sessions:
                return self._sessions[name]
            session = await self._create_persistent_session(name)
            if session:
                self._sessions[name] = session
            return session

    async def _create_persistent_session(self, name: str) -> Optional[ClientSession]:
        """Create session with transport context kept alive for reuse."""
        server_config = self.config.get().mcpServers.get(name)
        if not server_config:
            return None

        try:
            # Create transport and enter context (keep alive)
            if isinstance(server_config, HttpServerConfig):
                ctx = streamablehttp_client(server_config.url)
                read, write, http_session = await ctx.__aenter__()
                self._http_sessions[name] = http_session  # Store for proper cleanup
            elif isinstance(server_config, SseServerConfig):
                ctx = sse_client(server_config.url)
                read, write = await ctx.__aenter__()
            elif isinstance(server_config, StdioServerConfig):
                params = StdioServerParameters(
                    command=server_config.command,
                    args=server_config.args or [],
                    env=server_config.env,
                )
                log_file = self._log_file_for_session(name)
                ctx = stdio_client(params, errlog=log_file)
                read, write = await ctx.__aenter__()
            else:
                raise ValueError(f"Unknown server config type: {type(server_config)}")

            self._contexts[name] = ctx
            self._read_streams[name] = read  # Store for proper cleanup

            # Create and initialize session
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            return session
        except Exception as e:
            print(
                f"Warning: Failed to connect to the '{name}' MCP server: {e}",
                file=sys.stderr,
            )
            print(
                f"Tools from '{name}' will be unavailable (run with LLM_TOOLS_MCP_FULL_ERRORS=1) or see logs: {self.config.log_path}",
                file=sys.stderr,
            )
            if os.environ.get("LLM_TOOLS_MCP_FULL_ERRORS", None):
                print(traceback.format_exc(), file=sys.stderr)
            return None

    @asynccontextmanager
    async def _client_session_with_logging(self, name, read, write):
        """Legacy context manager for ephemeral sessions (used by get_tools_for)."""
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
                yield session
            except Exception as e:
                print(
                    f"Warning: Failed to connect to the '{name}' MCP server: {e}",
                    file=sys.stderr,
                )
                print(
                    f"Tools from '{name}' will be unavailable (run with LLM_TOOLS_MCP_FULL_ERRORS=1) or see logs: {self.config.log_path}",
                    file=sys.stderr,
                )
                if os.environ.get("LLM_TOOLS_MCP_FULL_ERRORS", None):
                    print(traceback.format_exc(), file=sys.stderr)
                yield None

    @asynccontextmanager
    async def _client_session(self, name: str):
        """Legacy context manager for ephemeral sessions (used by get_tools_for)."""
        server_config = self.config.get().mcpServers.get(name)
        if not server_config:
            raise ValueError(f"There is no such MCP server: {name}")
        if isinstance(server_config, SseServerConfig):
            async with sse_client(server_config.url) as (read, write):
                async with self._client_session_with_logging(
                    name, read, write
                ) as session:
                    yield session
        elif isinstance(server_config, HttpServerConfig):
            async with streamablehttp_client(server_config.url) as (read, write, _):
                async with self._client_session_with_logging(
                    name, read, write
                ) as session:
                    yield session
        elif isinstance(server_config, StdioServerConfig):
            params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args or [],
                env=server_config.env,
            )
            log_file = self._log_file_for_session(name)
            async with stdio_client(params, errlog=log_file) as (read, write):
                async with self._client_session_with_logging(
                    name, read, write
                ) as session:
                    yield session
        else:
            raise ValueError(f"Unknown server config type: {type(server_config)}")

    def _log_file_for_session(self, name: str) -> TextIO:
        log_file = (
            self.config.log_path.parent
            / "logs"
            / f"{name}-{uuid.uuid4()}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return open(log_file, "w")

    async def get_tools_for(self, name: str) -> ListToolsResult:
        """Get tools for a specific server (uses ephemeral session for discovery)."""
        async with self._client_session(name) as session:
            if session is None:
                return ListToolsResult(tools=[])
            return await session.list_tools()

    async def get_all_tools(self) -> tuple[dict[str, list[Tool]], bool]:
        """Fetch tools from all servers in parallel.

        Returns:
            Tuple of (tools_dict, had_errors) where had_errors is True if any server failed.
        """
        server_names = list(self.config.get().mcpServers.keys())
        had_errors = False

        async def fetch_tools(name: str) -> tuple[str, list[Tool], bool]:
            try:
                result = await self.get_tools_for(name)
                return (name, result.tools, False)
            except Exception as e:
                print(f"Warning: Failed to get tools from '{name}': {e}", file=sys.stderr)
                return (name, [], True)

        # Fetch all in parallel
        results = await asyncio.gather(*[fetch_tools(name) for name in server_names])
        tools_dict = {}
        for name, tools, error in results:
            tools_dict[name] = tools
            if error:
                had_errors = True
        return tools_dict, had_errors

    async def call_tool(self, server_name: str, name: str, /, **kwargs):
        """Call tool using cached persistent session."""
        session = await self._get_or_create_session(server_name)
        if session is None:
            return f"Error: Failed to call tool {name} from MCP server {server_name}"
        tool_result = await session.call_tool(name, kwargs)

        # Check for structuredContent first (MCP 2025-06-18 spec) - contains JSON data
        # Some servers (like Microsoft Learn MCP) may return structured data here
        if hasattr(tool_result, 'structuredContent') and tool_result.structuredContent:
            import json
            return json.dumps(tool_result.structuredContent)

        # Extract text from MCP content blocks (TextContent, ImageContent, etc.)
        # This ensures JSON responses are returned as actual JSON strings,
        # not Python repr of content objects like "[TextContent(type='text', text='...')]"
        # Handle both pydantic objects (hasattr) and dicts (from JSON deserialization)
        text_parts = []
        if tool_result.content:
            for block in tool_result.content:
                # Handle pydantic/dataclass objects with .text attribute
                if hasattr(block, 'text') and block.text:
                    text_parts.append(block.text)
                # Handle dict-style content blocks (from JSON deserialization)
                elif isinstance(block, dict) and block.get('text'):
                    text_parts.append(block['text'])
        return '\n'.join(text_parts) if text_parts else str(tool_result.content)

    async def close_all(self):
        """Properly cleanup all cached sessions and transports.

        Called from the persistent loop thread during cleanup.
        """
        # Close sessions first (copy to list to avoid dict modification during iteration)
        for name, session in list(self._sessions.items()):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass

        # Close read streams (anyio MemoryObjectReceiveStream - prevents async iterator warnings)
        for name, read_stream in list(self._read_streams.items()):
            try:
                if hasattr(read_stream, 'aclose'):
                    await read_stream.aclose()
            except Exception:
                pass

        # Close HTTP session objects (have async iterators that need aclose)
        for name, http_session in list(self._http_sessions.items()):
            try:
                if hasattr(http_session, 'aclose'):
                    await http_session.aclose()
            except Exception:
                pass

        # Then close transport contexts
        for name, ctx in list(self._contexts.items()):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass

        self._sessions.clear()
        self._read_streams.clear()
        self._http_sessions.clear()
        self._contexts.clear()

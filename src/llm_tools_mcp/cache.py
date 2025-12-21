"""Tool definition caching for llm-tools-mcp.

Caches tool schemas (name, description, inputSchema) to disk, allowing
subsequent startups to skip the MCP server connection for tool discovery.
Cache invalidates when the mcp.json config file changes (different hash).

Note: This only caches tool *schemas* for faster startup. Actual tool
*invocation* still connects to MCP servers via session persistence.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any


CACHE_DIR = Path.home() / ".cache" / "llm-tools-mcp"
CACHE_TTL = timedelta(hours=24)


def get_cache_key(config_content: str) -> str:
    """Generate a cache key by hashing the config content."""
    return hashlib.sha256(config_content.encode()).hexdigest()[:16]


def load_cached_tools(config_content: str) -> Optional[list[dict[str, Any]]]:
    """Load tools from cache if valid.

    Args:
        config_content: The raw content of mcp.json

    Returns:
        List of tool dicts if cache is valid, None otherwise.
        Each dict contains: server, name, description, schema
    """
    cache_key = get_cache_key(config_content)
    cache_file = CACHE_DIR / f"tools-{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        if datetime.now() - cached_at > CACHE_TTL:
            return None  # Expired
        return data["tools"]
    except Exception:
        return None


def save_tools_cache(config_content: str, tools: list[dict[str, Any]]) -> None:
    """Save tools to cache.

    Args:
        config_content: The raw content of mcp.json (used for cache key)
        tools: List of tool dicts with server, name, description, schema
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(config_content)
    cache_file = CACHE_DIR / f"tools-{cache_key}.json"

    data = {
        "cached_at": datetime.now().isoformat(),
        "tools": tools
    }
    cache_file.write_text(json.dumps(data, indent=2))


def clear_cache() -> None:
    """Clear all cached tool definitions."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("tools-*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass

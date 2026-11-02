from __future__ import annotations
import asyncio
import json
import os
import sys
import tempfile
import threading
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

ALLOWED_TOOLS = {"search"}

class SearchUnavailable (RuntimeError):
    pass

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

# parse raw duckduckgo response into search result objects
def parse_ddg_text (text: str):
    if not text or text.lstrip ().startswith ("No results"):
        return []
    out = []
    current_title = None
    current_url = None
    summary_lines = []

    def flush_item ():
        nonlocal current_title, current_url, summary_lines
        if current_title and current_url:
            snip = " ".join ("\n".join (summary_lines).split ())
            out.append (SearchResult (current_title, current_url, snip))
        current_title = None
        current_url = None
        summary_lines = []

    for raw_line in text.splitlines ():
        line = raw_line.strip ()
        parts = line.split (". ", 1)
        if len (parts) == 2 and parts [0].isdigit () and len (parts [0]) <= 3:
            flush_item ()
            current_title = parts [1].strip ()
        elif line.startswith ("URL:"):
            current_url = line [4 :].strip ().split () [0]
        elif line.startswith ("Summary:"):
            summary_lines.append (line [8 :].strip ())
        elif current_url:
            summary_lines.append (raw_line)

    flush_item ()
    return out

INJECTION_PHRASES = (
    "ignore all previous",
    "ignore previous",
    "ignore all prior",
    "ignore prior",
    "ignore all instructions",
    "ignore all prompts",
    "ignore prompt",
    "disregard",
    "you are now",
    "system prompt",
)

# clean search results to prevent prompt injection and control chars
def sanitize (results: List [SearchResult], max_chars: int = 350):
    clean = []
    for r in results:
        def c (s, n):
            if not s:
                s = ""
            tag_chars = []
            in_tag = False
            for ch in s:
                if ch == "<":
                    in_tag = True
                elif ch == ">":
                    in_tag = False
                elif not in_tag:
                    tag_chars.append (ch)
            cleaned = "".join (tag_chars)

            safe_chars = []
            for ch in cleaned:
                code = ord (ch)
                if (code >= 0 and code <= 8) or (code >= 11 and code <= 31) or ch == "`":
                    safe_chars.append (" ")
                else:
                    safe_chars.append (ch)
            text_str = "".join (safe_chars)

            for phrase in INJECTION_PHRASES:
                if phrase in text_str.lower ():
                    pos = 0
                    while True:
                        idx = text_str.lower ().find (phrase, pos)
                        if idx < 0:
                            break
                        text_str = text_str [: idx] + "[removed]" + text_str [idx + len (phrase) :]
                        pos = idx + len ("[removed]")
            return " ".join (text_str.split ()) [: n]

        title = c (r.title, 120)
        url = (r.url or "") [: 300]
        snippet = c (r.snippet, max_chars)
        clean.append (SearchResult (title, url, snippet))
    return clean

# base interface for search backends
class SearchBackend:
    def search (self, query: str, k: int = 3):
        raise NotImplementedError

    def close (self):
        pass

# mcp client for duckduckgo search tool
class MCPSearch (SearchBackend):
    def __init__ (self, command: str = "duckduckgo-mcp-server", args: Optional [List [str]] = None, timeout: float = 45.0):
        if command == "duckduckgo-mcp-server" and sys.platform == "win32":
            self.command = sys.executable
            extra = args or []
            self.args = ["-m", "duckduckgo_mcp_server.server"] + extra
        else:
            self.command = command
            self.args = args or []

        self.timeout = timeout
        self._loop = asyncio.new_event_loop ()
        self._thread = threading.Thread (target = self._loop.run_forever, daemon = True)
        self._thread.start ()
        self._stack: Optional [AsyncExitStack] = None
        self._session = None
        self._run (self._start ())

    # run coroutine safely on background loop
    def _run (self, coro):
        fut = asyncio.run_coroutine_threadsafe (coro, self._loop)
        return fut.result (self.timeout + 15)

    # initialize mcp stdio session
    async def _start (self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack ()
        params = StdioServerParameters (command = self.command, args = self.args)
        r, w = await self._stack.enter_async_context (stdio_client (params))
        self._session = await self._stack.enter_async_context (ClientSession (r, w))
        await self._session.initialize ()

        tools_resp = await self._session.list_tools ()
        names = set ()
        for t in tools_resp.tools:
            names.add (t.name)

        if "search" not in names:
            raise SearchUnavailable (f"MCP server exposes {names}; no 'search' tool.")

    # perform async search tool call
    async def _search (self, query: str, k: int):
        tool = "search"
        assert tool in ALLOWED_TOOLS

        call_coro = self._session.call_tool (tool, {"query": query, "max_results": k})
        res = await asyncio.wait_for (call_coro, self.timeout)

        if getattr (res, "is_error", None) or getattr (res, "isError", None):
            return []

        text_parts = []
        for c in res.content:
            text_parts.append (getattr (c, "text", ""))
        text = "\n".join (text_parts)

        parsed = parse_ddg_text (text)
        return parsed [: k]

    def search (self, query: str, k: int = 3):
        return self._run (self._search (query, k))

    # close active session and event loop
    def close (self):
        if self._stack:
            try:
                self._run (self._stack.aclose ())
            except Exception:
                pass
        self._loop.call_soon_threadsafe (self._loop.stop)

# mock search backend for offline testing
class StubSearch (SearchBackend):
    def __init__ (self, fn: Optional [Callable [[str], List [SearchResult]]] = None):
        self.fn = fn
        self.calls = 0

    def search (self, query, k = 3):
        self.calls = self.calls + 1
        if self.fn:
            return self.fn (query) [: k]

        words = " ".join (query.split () [: 12])
        results = []
        for i in range (1, k + 1):
            title = "Stub source " + str (i)
            url = "https://example.org/" + str (i)
            snip = "Reference text number " + str (i) + " says that " + words + " according to the stub index."
            results.append (SearchResult (title, url, snip))
        return results

# caches search results to disk for replay across models
class SnapshotSearch (SearchBackend):
    def __init__ (self, inner: Optional [SearchBackend], path: str, offline: bool = False):
        self.inner = inner
        self.path = path
        self.offline = offline
        self.data: Dict [str, list] = {}
        self.meta: Dict [str, str] = {}

        if os.path.exists (path):
            with open (path, encoding = "utf-8") as f:
                self.data = json.load (f)
            self.meta = self.data.pop ("__meta__", {})

    # flush data atomically to json file
    def _flush (self):
        d = os.path.dirname (os.path.abspath (self.path))
        os.makedirs (d, exist_ok = True)
        fd, tmp = tempfile.mkstemp (dir = d)
        merged = {}
        for k, v in self.data.items ():
            merged [k] = v
        merged ["__meta__"] = self.meta

        with os.fdopen (fd, "w", encoding = "utf-8") as f:
            json.dump (merged, f, ensure_ascii = False, indent = 1)
        os.replace (tmp, self.path)

    # memoize generated query for reuse
    def memo (self, key: str, fn):
        if key in self.meta:
            return self.meta [key]
        if self.offline:
            raise SearchUnavailable ("Offline mode and no stored query for " + repr (key))
        self.meta [key] = fn ()
        self._flush ()
        return self.meta [key]

    # retrieve search results from snapshot or inner backend
    def search (self, query, k = 3):
        if query in self.data:
            cached = []
            for r in self.data [query]:
                cached.append (SearchResult (**r))
            return cached [: k]

        if self.offline or self.inner is None:
            raise SearchUnavailable ("Offline mode and no snapshot for query: " + repr (query [: 80]))

        res = sanitize (self.inner.search (query, k))
        if res:
            res_dicts = []
            for r in res:
                res_dicts.append (asdict (r))
            self.data [query] = res_dicts
            self._flush ()
        return res

    def close (self):
        if self.inner:
            self.inner.close ()

# build simple keyword query from question and option text
def build_query (question: str, option_text: str, max_len: int = 200):
    q = " ".join (question.split ())
    full = " ".join ((q + " " + option_text).split ())
    return full [: max_len]

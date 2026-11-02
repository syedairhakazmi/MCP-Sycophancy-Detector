import os
import sys
import json
import asyncio
import tempfile
import threading
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

ALLOWED_TOOLS = {"search"}

class SearchUnavailable (RuntimeError):
    pass

# search result object
class SearchResult:
    def __init__ (self, title, url, snippet):
        self.title = title
        self.url = url
        self.snippet = snippet

    def __eq__ (self, other):
        if not isinstance (other, SearchResult):
            return False
        return self.title == other.title and self.url == other.url and self.snippet == other.snippet

    def __repr__ (self):
        return f"SearchResult(title={self.title!r}, url={self.url!r}, snippet={self.snippet!r})"

# parse raw duckduckgo response into search result objects
def parse_ddg_text (text):
    if not text or text.lstrip ().startswith ("No results"):
        return []
    
    out = []
    current_title = None
    current_url = None
    summary_lines = []
    
    for raw_line in text.splitlines ():
        line = raw_line.strip ()
        parts = line.split (". ", 1)
        
        if len (parts) == 2 and parts [0].isdigit () and len (parts [0]) <= 3:
            if current_title and current_url:
                snip = " ".join ("\n".join (summary_lines).split ())
                out.append (SearchResult (current_title, current_url, snip))
            current_title = parts [1].strip ()
            current_url = None
            summary_lines = []
        elif line.startswith ("URL:"):
            current_url = line [4 : ].strip ().split () [0]
        elif line.startswith ("Summary:"):
            summary_lines.append (line [8 : ].strip ())
        elif current_url:
            summary_lines.append (raw_line)
   
    if current_title and current_url:
        snip = " ".join ("\n".join (summary_lines).split ())
        out.append (SearchResult (current_title, current_url, snip))
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
def sanitize (results, max_chars = 350):
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
                        
                        text_str = text_str [ : idx] + "[removed]" + text_str [idx + len (phrase) : ]
                        pos = idx + len ("[removed]")
            
            return " ".join (text_str.split ()) [ : n]

        title = c (r.title, 120)
        url = (r.url or "") [ : 300]
        snippet = c (r.snippet, max_chars)
        clean.append (SearchResult (title, url, snippet))
    
    return clean

# base interface for search backends
class SearchBackend:
    def search (self, query, k = 3):
        raise NotImplementedError

    def close (self):
        pass

# mcp client for duckduckgo search tool
class MCPSearch (SearchBackend):
    def __init__ (self, command = "duckduckgo-mcp-server", args = None, timeout = 45.0):
        if command == "duckduckgo-mcp-server" and sys.platform == "win32":
            self.command = sys.executable
            extra = args or []
            self.args = ["-m", "duckduckgo_mcp_server"] + extra
        else:
            self.command = command
            self.args = args or []

        self.timeout = timeout
        self._loop = asyncio.new_event_loop ()
        self._thread = threading.Thread (target = self._loop.run_forever, daemon = True)
        self._thread.start ()
        self._stack = None
        self._session = None
        self._run (self._start ())

    def _run (self, coro):
        fut = asyncio.run_coroutine_threadsafe (coro, self._loop)
        return fut.result ()

    async def _start (self):
        self._stack = AsyncExitStack ()
        server_params = StdioServerParameters (command = self.command, args = self.args)
        read_stream, write_stream = await self._stack.enter_async_context (stdio_client (server_params))
        self._session = await self._stack.enter_async_context (ClientSession (read_stream, write_stream))
        await self._session.initialize ()

        tools_resp = await self._session.list_tools ()
        names = set ()
        
        for t in tools_resp.tools:
            names.add (t.name)

        if "search" not in names:
            raise SearchUnavailable (f"MCP server exposes {names}; no 'search' tool.")

    # perform async search tool call
    async def _search (self, query, k):
        tool = "search"
        assert tool in ALLOWED_TOOLS

        call_coro = self._session.call_tool (tool, {"query": query, "max_results": k})
        res = await asyncio.wait_for (call_coro, self.timeout)
        text_parts = []
        for block in res.content:
            text_val = getattr (block, "text", None)
            if text_val is not None:
                text_parts.append (text_val)
            else:
                text_parts.append (str (block))
        text = "\n".join (text_parts)

        parsed = parse_ddg_text (text)
        return parsed [ : k]

    def search (self, query, k = 3):
        return self._run (self._search (query, k))

    # close active session and event loop
    def close (self):
        try:
            if self._stack:
                self._run (self._stack.aclose ())
        finally:
            self._loop.call_soon_threadsafe (self._loop.stop)
            self._thread.join (timeout = 2.0)

# mock search backend for offline testing
class StubSearch (SearchBackend):
    def __init__ (self, fn = None):
        self.fn = fn
        self.calls = 0

    def search (self, query, k = 3):
        self.calls = self.calls + 1
        
        if self.fn:
            return self.fn (query) [ : k]

        words = " ".join (query.split () [ : 12])
        results = []
        
        for i in range (1, k + 1):
            title = "Stub source " + str (i)
            url = "https://example.org/" + str (i)
            snip = "Reference text number " + str (i) + " says that " + words + " according to the stub index."
            results.append (SearchResult (title, url, snip))
        
        return results

# caches search results to disk for replay across models
class SnapshotSearch (SearchBackend):
    def __init__ (self, inner = None, path = "", offline = False):
        self.inner = inner
        self.path = path
        self.offline = offline
        self.data = {}
        self.meta = {}

        if os.path.exists (path):
            with open (path, encoding = "utf-8") as f:
                loaded = json.load (f)
                self.meta = loaded.pop ("__meta__", {})
                self.data = loaded

    def _flush (self):
        d = os.path.dirname (os.path.abspath (self.path))
        os.makedirs (d, exist_ok = True)
        fd, tmp = tempfile.mkstemp (dir = d)
        merged = {}
        
        for k, v in self.data.items ():
            merged [k] = v
        
        merged ["__meta__"] = self.meta

        with os.fdopen (fd, "w", encoding = "utf-8") as f:
            json.dump (merged, f, indent = 1, sort_keys = True)
        os.replace (tmp, self.path)

    # memoize generated query for reuse
    def memo (self, key, fn):
        if key in self.meta:
            return self.meta [key]
        
        if self.offline:
            raise SearchUnavailable ("Offline mode and no stored query for " + repr (key))
        
        self.meta [key] = fn ()
        self._flush ()
        return self.meta [key]

    def search (self, query, k = 3):
        if query in self.data:
            cached = []
            for r in self.data [query]:
                cached.append (SearchResult (**r))
            return cached [ : k]

        if self.offline or self.inner is None:
            raise SearchUnavailable ("Offline mode and no snapshot for query: " + repr (query [ : 80]))

        res = sanitize (self.inner.search (query, k))
        
        if res:
            res_dicts = []
            for r in res:
                res_dicts.append ({"title": r.title, "url": r.url, "snippet": r.snippet})
            self.data [query] = res_dicts
            self._flush ()
        return res [ : k]

    def close (self):
        if self.inner:
            self.inner.close ()

# build simple keyword query from question and option text
def build_query (question, option_text, max_len = 200):
    q = " ".join (question.split ())
    full = " ".join ((q + " " + option_text).split ())
    
    return full [ : max_len]
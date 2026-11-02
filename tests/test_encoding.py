# Regression: Windows defaults to cp1252, which crashed on model text like U+2212 / U+2265.
import os
import sys
import subprocess

ROOT = os.path.dirname (os.path.dirname (os.path.abspath (__file__)))

def test_every_text_open_declares_encoding ():
    bad = []
    for fn in sorted (os.listdir (os.path.join (ROOT, "syc"))):
        if not fn.endswith (".py"):
            continue
        src = open (os.path.join (ROOT, "syc", fn), encoding = "utf-8").read ()
        for line in src.splitlines ():
            line_str = line.strip ()
            if line_str.startswith ("#"):
                continue
            if "open (" in line_str or "open(" in line_str or "os.fdopen (" in line_str or "os.fdopen(" in line_str:
                if "encoding" not in line_str and '"rb"' not in line_str and '"wb"' not in line_str:
                    bad.append (f"{fn}: {line_str [ : 70]}")
    assert not bad, "open () without encoding= (breaks on Windows):\n" + "\n".join (bad)

def test_unicode_survives_a_non_utf8_default_locale (tmp_path):
    script_lines = [
        "import json, os, sys",
        f"sys.path.insert (0, {ROOT!r})",
        "from syc.llm import Tracer",
        "from syc.search import SnapshotSearch, StubSearch, SearchResult",
        f"t = Tracer ({str (tmp_path / 'trace.jsonl')!r}, 'r1')",
        "t.log (role = 'x', output = 'p \\u2265 0.05 and \\u2212 3')",
        "def _stub_fn (q): return [SearchResult ('t\\u2212', 'u', 'snippet \\u2265 \\u201cquoted\\u201d')]",
        f"s = SnapshotSearch (StubSearch (_stub_fn), {str (tmp_path / 'snap.json')!r})",
        "assert '\\u2265' in s.search ('q', 1) [0].snippet",
        f"assert '\\u2265' in SnapshotSearch (None, {str (tmp_path / 'snap.json')!r}, offline = True).search ('q', 1) [0].snippet",
        "print ('OK')",
    ]
    script = "\n".join (script_lines)
    env = dict (os.environ, PYTHONUTF8 = "0", PYTHONCOERCECLOCALE = "0", LC_ALL = "C", LANG = "C")
    r = subprocess.run ([sys.executable, "-c", script], env = env, capture_output = True, text = True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr [-600 : ]
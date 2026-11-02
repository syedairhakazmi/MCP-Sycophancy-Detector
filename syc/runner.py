from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from typing import List
from . import __version__, utf8_stdout
from .items import PAPER_SUBJECTS, check_items, load_bank, load_mmlu
from .llm import (
    BudgetExceeded,
    CallBudget,
    DiskCache,
    Instrumented,
    LLMError,
    LLMFatal,
    RateLimitExhausted,
    Tracer,
    get_llm,
)
from .pressure import ALL_STYLES, DEFAULT_STYLES, EVIDENCE_STYLE
from .schemas import Item
from .search import MCPSearch, SnapshotSearch, StubSearch
from .steps import Ctx, build_graph, run_graph, run_loop

# run git command to get commit hash
def _git (cmd: List [str], cwd = None):
    try:
        args = ["git"] + cmd
        out = subprocess.check_output (args, cwd = cwd, stderr = subprocess.DEVNULL, text = True)
        return out.strip ()
    except Exception:
        return ""

# add common command line arguments
def add_common (p: argparse.ArgumentParser):
    p.add_argument ("--source", choices = ["mmlu", "bank", "both"], default = "both")
    p.add_argument ("--base-dir", default = "third_party/kasprova")
    p.add_argument ("--mmlu-csv", default = None)
    p.add_argument ("--bank", default = "data/question_bank.json")
    p.add_argument ("--subjects", nargs = "+", default = PAPER_SUBJECTS)
    p.add_argument ("--n-per-subject", type = int, default = None)
    default_styles_str = ",".join (DEFAULT_STYLES)
    p.add_argument ("--styles", default = default_styles_str)
    p.add_argument ("--rounds", type = int, default = 3)
    p.add_argument ("--no-genuine-subjects", nargs = "*", default = ["elementary_mathematics"])
    p.add_argument ("--seed", type = int, default = 42)

# load questions according to arguments
def build_items (a):
    items: List [Item] = []
    if a.source == "mmlu" or a.source == "both":
        if a.mmlu_csv:
            csv_path = a.mmlu_csv
        else:
            csv_path = os.path.join (a.base_dir, "data_for_dss.csv")

        if not os.path.exists (csv_path):
            sys.exit (f"MMLU csv not found at {csv_path}. Run scripts/setup_base.sh first (or pass --mmlu-csv).")
        mmlu_items = load_mmlu (csv_path, a.subjects, a.n_per_subject, a.seed)
        items.extend (mmlu_items)

    if a.source == "bank" or a.source == "both":
        bank_items = load_bank (a.bank)
        items.extend (bank_items)

    check_items (items)
    return items

# build schedule pairing each question with pressure styles
def schedule (items: List [Item], styles: List [str], skip_genuine_subjects):
    out = []
    skip_set = set (skip_genuine_subjects or [])
    for it in items:
        for st in styles:
            if st == EVIDENCE_STYLE and it.subject in skip_set:
                continue
            out.append ((it, st))
    return out

# calculate and print call estimation plan
def cmd_plan (a):
    items = build_items (a)
    styles = a.styles.split (",")
    sched = schedule (items, styles, a.no_genuine_subjects)
    r_val = a.rounds

    n_gen = 0
    for _, s in sched:
        if s == EVIDENCE_STYLE:
            n_gen = n_gen + 1

    shared_init = len (items)
    worst = shared_init + len (sched) * 3 * r_val
    typical = shared_init + len (sched) * r_val * 1.3 + n_gen * r_val

    print (f"items={len (items)}  styles={styles}  rounds={r_val}")
    print (f"conversations (per answerer model) = {len (sched)}")
    print (f"LLM calls per answerer model: typical ~{int (typical)} (detector on flips only), worst case {worst}")
    print (f"MCP searches: <= {n_gen * r_val} (cached per query)")
    print ("Tip: check your Groq free-tier requests/minute and requests/day, then size --n-per-subject.")

# instantiate search backend based on mode
def make_search (mode: str, snapshot: str):
    if mode == "mcp":
        return SnapshotSearch (MCPSearch (), snapshot, offline = False)
    if mode == "stub":
        return SnapshotSearch (StubSearch (), snapshot, offline = False)
    if mode == "offline":
        return SnapshotSearch (None, snapshot, offline = True)
    if mode == "none":
        return None
    raise ValueError (mode)

# execute full experiment run across schedule
def cmd_run (a):
    items = build_items (a)
    styles = a.styles.split (",")
    for s in styles:
        assert s in ALL_STYLES, "unknown style " + str (s)

    os.makedirs (a.out, exist_ok = True)
    run_id = uuid.uuid4 ().hex [: 8]
    cache = DiskCache (os.path.join (a.out, "llm_cache.sqlite"))
    tracer = Tracer (os.path.join (a.out, "trace.jsonl"), run_id)
    budget = CallBudget (a.max_calls)

    lookup = {}
    for it in items:
        q_first_line = it.question.split ("\n") [0]
        lookup [q_first_line] = it.gold

    kw = {
        "reasoning_effort": a.reasoning_effort,
        "load_in_4bit": a.load_in_4bit,
        "max_wait": a.max_wait,
        "mock_kwargs": {"gold_lookup": lookup.get},
    }

    _models = {}

    def mk (spec, role, max_tokens):
        if spec not in _models:
            _models [spec] = get_llm (spec, **kw)
        return Instrumented (_models [spec], role, cache, tracer, budget, max_tokens = max_tokens)

    answerer = mk (a.answerer, "answerer", 300)
    if a.pressure:
        press_spec = a.pressure
    else:
        press_spec = a.answerer
    pressure = mk (press_spec, "pressure", 250)

    if a.detector:
        det_spec = a.detector
    else:
        det_spec = a.answerer
    detector = mk (det_spec, "detector", 300)

    if a.snapshot:
        snapshot = a.snapshot
    else:
        snapshot = os.path.join (a.out, "search_snapshots.json")
    search = make_search (a.search, snapshot)

    ctx = Ctx (
        answerer,
        pressure,
        detector,
        search,
        rounds = a.rounds,
        detect_mode = a.detect,
        detector_input = a.detector_input,
        k_results = a.k_results,
        seed = a.seed,
        query_mode = a.query_mode,
    )
    if a.engine == "langgraph":
        graph = build_graph (ctx)
    else:
        graph = None

    out_path = os.path.join (a.out, "conversations.jsonl")
    done = set ()
    if os.path.exists (out_path):
        with open (out_path, encoding = "utf-8") as f:
            for line in f:
                try:
                    r0 = json.loads (line)
                    status_val = str (r0.get ("status", ""))
                    if not status_val.startswith ("aborted:llm_error"):
                        done.add (r0 ["conv_id"])
                except Exception:
                    pass

    cfg = {
        "version": __version__,
        "args": vars (a),
        "run_id": run_id,
        "python": platform.python_version (),
        "code_commit": _git (["rev-parse", "HEAD"]),
        "base_repo_commit": _git (["rev-parse", "HEAD"], a.base_dir),
        "started": time.strftime ("%Y-%m-%d %H:%M:%S"),
    }
    with open (os.path.join (a.out, f"config_{run_id}.json"), "w", encoding = "utf-8") as f:
        json.dump (cfg, f, indent = 1, default = str)

    sched = schedule (items, styles, a.no_genuine_subjects)
    print (f"[{run_id}] {len (sched)} conversations scheduled, {len (done)} already done (resume).")
    t0 = time.time ()

    try:
        n = 1
        for item, style in sched:
            cid = item.id + "|" + style + "|" + answerer.name + "|" + str (a.seed)
            if cid in done:
                n = n + 1
                continue

            try:
                if graph:
                    rec = run_graph (graph, ctx, item, style)
                else:
                    rec = run_loop (ctx, item, style)
            except (BudgetExceeded, RateLimitExhausted, LLMFatal):
                raise
            except LLMError as e:
                err_text = str (e) [: 160]
                print (f"  ! {cid}: {type (e).__name__}: {err_text} (recorded as aborted; will be retried on resume)", flush = True)
                rec = {
                    "conv_id": cid,
                    "item": item.meta (),
                    "style": style,
                    "answerer": answerer.name,
                    "pressure_model": pressure.name,
                    "detector_model": detector.name,
                    "detector_input": a.detector_input,
                    "detect_mode": a.detect,
                    "engine": a.engine,
                    "seed": a.seed,
                    "n_rounds": a.rounds,
                    "status": "aborted:llm_error",
                    "status_detail": {"reason": type (e).__name__ + ": " + str (e) [: 300]},
                    "s0": None,
                    "s0_valid": False,
                    "initial_correct": None,
                    "rounds": [],
                }

            with open (out_path, "a", encoding = "utf-8") as f:
                f.write (json.dumps (rec, ensure_ascii = False) + "\n")

            if n % 5 == 0 or n == len (sched):
                elapsed = int (time.time () - t0)
                print (f"  {n}/{len (sched)}  new LLM calls={budget.used}  {elapsed}s", flush = True)
            n = n + 1

    except BudgetExceeded as e:
        print (f"STOPPED: {e} (re-run the same command to resume)")
    except LLMFatal as e:
        print (f"STOPPED: {e}")
    except RateLimitExhausted as e:
        print (f"STOPPED: {e}\n  Almost certainly a daily limit. Finished conversations and LLM calls are saved: re-run the SAME command later (or with another model) and it resumes.")
    finally:
        if search is not None:
            search.close ()
    print (f"done -> {out_path}")

def main (argv = None):
    utf8_stdout ()
    ap = argparse.ArgumentParser (prog = "syc.runner")
    sub = ap.add_subparsers (dest = "cmd", required = True)

    p1 = sub.add_parser ("plan")
    add_common (p1)

    p2 = sub.add_parser ("run")
    add_common (p2)
    p2.add_argument ("--answerer", required = True)
    p2.add_argument ("--pressure", default = None)
    p2.add_argument ("--detector", default = None)
    p2.add_argument ("--detect", choices = ["flips", "all", "none"], default = "flips")
    p2.add_argument ("--detector-input", choices = ["verifier", "transcript"], default = "verifier")
    p2.add_argument ("--search", choices = ["mcp", "stub", "offline", "none"], default = "mcp")
    p2.add_argument ("--snapshot", default = None)
    p2.add_argument ("--k-results", type = int, default = 5)
    p2.add_argument ("--query-mode", choices = ["llm", "template"], default = "llm")
    p2.add_argument ("--max-wait", type = float, default = 300.0)
    p2.add_argument ("--engine", choices = ["loop", "langgraph"], default = "langgraph")
    p2.add_argument ("--max-calls", type = int, default = None)
    p2.add_argument ("--reasoning-effort", default = "low")
    p2.add_argument ("--load-in-4bit", action = "store_true")
    p2.add_argument ("--out", required = True)

    a = ap.parse_args (argv)
    commands = {"plan": cmd_plan, "run": cmd_run}
    commands [a.cmd] (a)

if __name__ == "__main__":
    main ()

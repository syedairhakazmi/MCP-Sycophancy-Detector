import os
import uuid
import json
import argparse
from . import utf8_stdout
from .schemas import Item
from .steps import silver_label
from collections import Counter
from .search import SearchResult
from .records import load_records
from .agents import detector_turn
from .verifier import verify_message
from .llm import CallBudget, DiskCache, Instrumented, Tracer, get_llm

def reverify_records (records):
    # Check the evidence and update the records
    stats = Counter ()

    for rec in records:
        gold = rec ["item"]["gold"]

        for x in rec ["rounds"]:
            ev = x.get ("evidence")

            if not ev or not ev.get ("results"):
                continue

            search_results = []

            for r in ev ["results"]:
                search_results.append (SearchResult (**r))

            message = x ["pressure_message"] or ""
            ver = verify_message (message, search_results)

            if ver ["report"] != ev.get ("report") or ver ["verified"] != ev.get ("verified"):
                ev ["verified"] = ver ["verified"]
                ev ["citations"] = ver ["citations"]
                ev ["report"] = ver ["report"]

                x ["reverified"] = True
                stats ["rounds_reverified"] += 1

                if x ["flip"] and x.get ("detector") and rec ["detector_input"] == "verifier":
                    x ["detector_stale"] = True
                    stats ["detector_stale"] += 1

            new = silver_label (rec ["style"], x ["flip"], x ["stance"], x ["target"], gold, bool (ev.get ("verified")))

            if new != x ["silver"]:
                key = "silver " + str (x ["silver"]) + "->" + str (new)
                stats [key] += 1
                x ["silver"] = new

    return stats

def redetect (records, llm):
    n = 0

    for rec in records:
        it = rec ["item"]

        item = Item (it ["id"], it ["source"], it ["tier"], it ["question"], it ["options"], it ["gold"], it ["subject"])

        for x in rec ["rounds"]:
            stale = x.pop ("detector_stale", False)

            if stale:
                evidence = x.get ("evidence") or {}
                report = evidence.get ("report", "")

                d = detector_turn (llm, item, x ["stance_before"], x ["stance"], x ["pressure_message"], x ["answerer_reply"], report, rec ["detector_input"])

                if d ["valid"]:
                    d ["flipped_agrees"] = d ["verdict"]["flipped"] == x ["flip"]

                x ["detector"] = d
                x ["detector_redone"] = True

                n += 1

    return n

def main (argv = None):
    utf8_stdout ()

    ap = argparse.ArgumentParser (prog = "syc.reverify")

    ap.add_argument ("--runs", nargs = "+", required = True)
    ap.add_argument ("--out", required = True, help = "new conversations.jsonl (the input files are never modified)")
    ap.add_argument ("--detector", default = None, help = "model spec used to re-ask the Detector on changed events")
    ap.add_argument ("--max-calls", type = int, default = None)
    ap.add_argument ("--reasoning-effort", default = "low")

    a = ap.parse_args (argv)

    records = load_records (a.runs)
    stats = reverify_records (records)

    print ("records (deduplicated): " + str (len (records)))

    for k, v in sorted (stats.items ()):
        print ("  " + str (k) + ": " + str (v))

    stale = stats.get ("detector_stale", 0)

    out_dir = os.path.dirname (os.path.abspath (a.out))
    os.makedirs (out_dir, exist_ok = True)

    if a.detector and stale:
        inner = get_llm (a.detector, reasoning_effort = a.reasoning_effort)

        cache_path = os.path.join (out_dir, "llm_cache.sqlite")
        trace_path = os.path.join (out_dir, "trace.jsonl")

        llm = Instrumented (inner, "detector_redo", DiskCache (cache_path), Tracer (trace_path, uuid.uuid4 ().hex [ : 8]), CallBudget (a.max_calls), max_tokens = 300)

        count = redetect (records, llm)

        print ("re-asking the Detector for " + str (count) + " stance changes")

    elif stale:
        print ("WARNING: " + str (stale) + " Detector verdicts were computed from the OLD verifier report and are flagged 'detector_stale'. Re-run with --detector <model> to refresh them.")

    with open (a.out, "w", encoding = "utf-8") as f:
        for r in records:
            line = json.dumps (r, ensure_ascii = False)
            f.write (line + "\n")

    print ("written -> " + a.out)

if __name__ == "__main__":
    main ()
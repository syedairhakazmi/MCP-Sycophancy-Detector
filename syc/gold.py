import os
import csv
import json
import random
import argparse
from . import utf8_stdout
from .stats import cohen_kappa
from collections import defaultdict

# iterate over all flips in conversation files
def iter_flips (paths):
    out = []
    for p in paths:
        with open (p, encoding = "utf-8") as f:
            for line in f:
                rec = json.loads (line)
                
                for r in rec ["rounds"]:
                    if r ["flip"]:
                        out.append ((rec, r))
    return out

# format unique event id for a round flip
def event_id (rec, r):
    return rec ["conv_id"] + "#r" + str (r ["round"])

# export blind flip sample for human rating
def cmd_export (a):
    flips = list (iter_flips (a.runs))
    by_style = defaultdict (list)
    
    for rec, r in flips:
        by_style [rec ["style"]].append ((rec, r))

    rng = random.Random (a.seed)
    chosen = []
    
    for style, evs in sorted (by_style.items ()):
        rng.shuffle (evs)
        chosen.extend (evs [ : a.per_style])

    # shuffle chosen events so raters are blinded to style
    rng.shuffle (chosen)

    target_dir = os.path.dirname (os.path.abspath (a.out))
    if not target_dir:
        target_dir = "."
    os.makedirs (target_dir, exist_ok = True)

    with open (a.out, "w", newline = "", encoding = "utf-8-sig") as f:
        w = csv.writer (f)
        w.writerow ([
            "event_id", "question", "options",
            "stance_before", "stance_after",
            "peer_message", "retrieved_sources",
            "answerer_reply", "rater1", "rater2", "adjudicated",
        ])

        for rec, r in chosen:
            item = rec ["item"]
            ev = r.get ("evidence")
            
            if not ev:
                ev = {}

            src_list = []
            ev_results = ev.get ("results")
            
            if not ev_results:
                ev_results = []
            i = 1
            
            for x in ev_results:
                src_list.append ("[" + str (i) + "] " + str (x ["url"]) + " :: " + str (x ["snippet"]))
                i = i + 1
            srcs = " || ".join (src_list)

            opt_list = []
            for k, v in sorted (item ["options"].items ()):
                opt_list.append (str (k) + ". " + str (v))
            opts = " | ".join (opt_list)

            w.writerow ([
                event_id (rec, r), item ["question"], opts,
                r ["stance_before"], r ["stance"],
                r ["pressure_message"], srcs, r ["answerer_reply"], "", "", ""])

    # save ground truth key in separate file
    key = {}
    for rec, r in chosen:
        eid = event_id (rec, r)
        key [eid] = {
            "style": rec ["style"],
            "silver": r ["silver"],
            "item": rec ["item"],
        }

    with open (a.out + ".key.json", "w", encoding = "utf-8") as f:
        json.dump (key, f, indent = 1)

    print ("exported " + str (len (chosen)) + " flip events (" + str (a.per_style) + " per style max) -> " + str (a.out))
    print ("Raters fill 'rater1' / 'rater2' with EG or SY. Do NOT share the .key.json file with raters.")

# import human labeled csv and compute agreement
def cmd_import (a):
    with open (a.csv, encoding = "utf-8-sig", newline = "") as f:
        rows = list (csv.DictReader (f))

    r1 = []
    r2 = []
    
    for r in rows:
        r1.append (r ["rater1"].strip ().upper ())
        r2.append (r ["rater2"].strip ().upper ())

    ok = []
    for i in range (len (rows)):
        if (r1 [i] == "EG" or r1 [i] == "SY") and (r2 [i] == "EG" or r2 [i] == "SY"):
            ok.append (i)

    r1_ok = []
    r2_ok = []
    
    for i in ok:
        r1_ok.append (r1 [i])
        r2_ok.append (r2 [i])

    kappa = cohen_kappa (r1_ok, r2_ok)

    same_count = 0
    for i in ok:
        if r1 [i] == r2 [i]:
            same_count = same_count + 1

    if len (ok) > 0:
        agree = same_count / len (ok)
    else:
        agree = 0.0

    print (f"labelled by both: {len (ok)}/{len (rows)}  raw agreement = {agree:.3f}  Cohen's kappa={kappa:.3f}")

    gold = {}
    unresolved = 0
    
    for i in ok:
        adj = rows [i]["adjudicated"].strip ().upper ()
        
        if r1 [i] == r2 [i]:
            gold [rows [i]["event_id"]] = r1 [i]
        elif adj == "EG" or adj == "SY":
            gold [rows [i]["event_id"]] = adj
        else:
            unresolved = unresolved + 1

    target_dir = os.path.dirname (os.path.abspath (a.out))
    
    if not target_dir:
        target_dir = "."
    os.makedirs (target_dir, exist_ok = True)

    with open (a.out, "w", encoding = "utf-8") as f:
        for k, v in gold.items ():
            record = {"event_id": k, "gold": v}
            f.write (json.dumps (record) + "\n")

    print (f"gold labels written: {len (gold)}  (disagreements still needing adjudication: {unresolved}) -> {a.out}")

def main (argv = None):
    utf8_stdout ()
    ap = argparse.ArgumentParser (prog = "syc.gold")
    sub = ap.add_subparsers (dest = "cmd", required = True)

    e = sub.add_parser ("export")
    e.add_argument ("--runs", nargs = "+", required = True, help = "conversations.jsonl files")
    e.add_argument ("--per-style", type = int, default = 25)
    e.add_argument ("--seed", type = int, default = 7)
    e.add_argument ("--out", default = "gold/gold_sample.csv")

    i = sub.add_parser ("import")
    i.add_argument ("--csv", required = True)
    i.add_argument ("--out", default = "gold/gold.jsonl")

    a = ap.parse_args (argv)
    commands = {"export": cmd_export, "import": cmd_import}
    commands [a.cmd] (a)

if __name__ == "__main__":
    main ()
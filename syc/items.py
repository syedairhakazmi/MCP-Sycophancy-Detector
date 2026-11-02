from __future__ import annotations
import json
import random
from typing import Iterable, List, Optional
import pandas as pd
from .schemas import Item

PAPER_SUBJECTS = [
    "elementary_mathematics",
    "professional_law",
    "machine_learning",
    "business_ethics",
    "high_school_biology",
]

# load mmlu questions from csv file
def load_mmlu (csv_path: str, subjects: Optional [Iterable [str]] = None, n_per_subject: Optional [int] = None, seed: int = 42):
    df = pd.read_csv (csv_path)
    if subjects:
        subjects_list = list (subjects)
    else:
        subjects_list = PAPER_SUBJECTS

    items = []
    rng = random.Random (seed)

    for subj in subjects_list:
        rows = []
        for i, r in df.iterrows ():
            if r ["subject"] == subj:
                rows.append ((i, r))

        if n_per_subject is not None and n_per_subject < len (rows):
            rows = rng.sample (rows, n_per_subject)
            rows.sort (key = lambda x: x [0])

        for i, r in rows:
            opts = {}
            for letter in "ABCD":
                opts [letter] = str (r ["choice_" + letter])

            item_obj = Item (
                id = "mmlu-" + str (i),
                source = "mmlu",
                tier = "mmlu",
                subject = subj,
                question = str (r ["question"]).strip (),
                options = opts,
                gold = str (r ["answer_letter"]).strip ().upper (),
            )
            items.append (item_obj)

    return items

# load question bank from json file
def load_bank (path: str):
    with open (path, encoding = "utf-8") as f:
        raw = json.load (f)

    items = []
    for r in raw:
        item_obj = Item (
            id = r ["id"],
            source = "bank",
            tier = r ["tier"],
            question = r ["question"],
            options = r ["options"],
            gold = r.get ("gold"),
            subject = r.get ("subject", r ["tier"]),
            note = r.get ("note", ""),
        )
        items.append (item_obj)

    return items

# validate list of questions
def check_items (items: List [Item]):
    ids = set ()
    for it in items:
        assert it.id not in ids, "duplicate id " + str (it.id)
        ids.add (it.id)
        assert len (it.options) >= 2, it.id
        if it.gold is not None:
            assert it.gold in it.options, str (it.id) + ": gold " + str (it.gold) + " not in options"
        if it.tier == "opinion":
            assert it.gold is None, str (it.id) + ": opinion items must not have a gold answer"

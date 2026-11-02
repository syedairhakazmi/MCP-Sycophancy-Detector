from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

# data class to store question item details
@dataclass
class Item:
    id: str
    source: str
    tier: str
    question: str
    options: Dict [str, str]
    gold: Optional [str] = None
    subject: str = ""
    note: str = ""

    # list sorted option keys
    @property
    def letters (self):
        return sorted (self.options)

    # format question and all options into string
    def render (self):
        opt_lines = []
        for k, v in sorted (self.options.items ()):
            opt_lines.append (f"{k}. {v}")
        opts = "\n".join (opt_lines)
        return f"{self.question}\n{opts}"

    # return metadata as dictionary
    def meta (self):
        return {
            "id": self.id,
            "source": self.source,
            "tier": self.tier,
            "subject": self.subject,
            "gold": self.gold,
            "question": self.question,
            "options": self.options,
        }

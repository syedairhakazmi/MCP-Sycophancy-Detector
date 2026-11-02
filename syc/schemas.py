# store question item details
class Item:
    def __init__ (self, id, source, tier, question, options, gold = None, subject = "", note = ""):
        self.id = id
        self.source = source
        self.tier = tier
        self.question = question
        self.options = options
        self.gold = gold
        self.subject = subject
        self.note = note

    def __eq__ (self, other):
        if not isinstance (other, Item):
            return False
        return (
            self.id == other.id
            and self.source == other.source
            and self.tier == other.tier
            and self.question == other.question
            and self.options == other.options
            and self.gold == other.gold
            and self.subject == other.subject
            and self.note == other.note
        )

    def __repr__ (self):
        return f"Item(id={self.id!r}, source={self.source!r}, tier={self.tier!r}, question={self.question!r}, options={self.options!r}, gold={self.gold!r}, subject={self.subject!r}, note={self.note!r})"

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
            "id": self.id, "source": self.source,
            "tier": self.tier, "subject": self.subject,
            "gold": self.gold, "question": self.question,
            "options": self.options,
        }
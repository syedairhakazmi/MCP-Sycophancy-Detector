import json

def rec_key (rec):
    return (
        rec ["conv_id"],
        rec.get ("detector_model"),
        rec.get ("detector_input"),
        rec.get ("n_rounds")
    )

def load_records (paths):
    last = []

    for p in paths:
        with open (p, encoding = "utf-8") as f:
            for line in f:
                if line.strip ():
                    rec = json.loads (line)
                    key = rec_key (rec)

                    found = False
                    i = 0

                    while i < len (last):
                        if rec_key (last [i]) == key:
                            last [i] = rec
                            found = True
                            break

                        i += 1

                    if not found:
                        last.append (rec)

    return last
import math
from collections import Counter

# calculate point estimate and wilson confidence interval
def wilson (k, n, z = 1.96):
    if n == 0:
        return (float ("nan"), float ("nan"), float ("nan"))
    
    p = k / n
    d = 1.0 + (z * z) / n
    c = (p + (z * z) / (2.0 * n)) / d
    h = (z * math.sqrt (p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / d
    
    return (p, max (0.0, c - h), min (1.0, c + h))

# calculate exact mcnemar test p-value
def mcnemar_exact (b, c):
    n = b + c
    
    if n == 0:
        return 1.0
    
    k = min (b, c)
    tail_sum = 0
    
    for i in range (0, k + 1):
        tail_sum = tail_sum + math.comb (n, i)
    tail = tail_sum / (2 ** n)
    
    return min (1.0, 2.0 * tail)

def _pval_val (kv):
    return kv [1]

# holm-bonferroni adjustment for multiple testing p-values
def holm (pvals):
    items = sorted (pvals.items (), key = _pval_val)
    m = len (items)
    out = {}
    running = 0.0
    i = 0
    
    for k, p in items:
        adjusted = (m - i) * p
        step_val = min (1.0, adjusted)
        running = max (running, step_val)
        out [k] = running
        i = i + 1
    return out

# calculate cohen kappa score for agreement
def cohen_kappa (a, b):
    assert len (a) == len (b)
    n = len (a)
    
    if n == 0:
        return float ("nan")

    matches = 0
    for i in range (len (a)):
        if a [i] == b [i]:
            matches = matches + 1
    po = matches / n

    ca = Counter (a)
    cb = Counter (b)
    all_keys = set (ca).union (set (cb))

    expected_sum = 0
    for l in all_keys:
        expected_sum = expected_sum + ca [l] * cb [l]
    pe = expected_sum / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)

# calculate precision, recall, and f1 scores
def prf (y_true, y_pred, labels):
    out = {}
    for l in labels:
        tp = 0
        fp = 0
        fn = 0
        for i in range (len (y_true)):
            t = y_true [i]
            p = y_pred [i]
            if t == l and p == l:
                tp = tp + 1
            if t != l and p == l:
                fp = fp + 1
            if t == l and p != l:
                fn = fn + 1

        if tp + fp > 0:
            pr = tp / (tp + fp)
        else:
            pr = float ("nan")

        if tp + fn > 0:
            rc = tp / (tp + fn)
        else:
            rc = float ("nan")

        if pr == pr and rc == rc and (pr + rc) > 0:
            f1 = 2.0 * pr * rc / (pr + rc)
        else:
            f1 = float ("nan")

        out [l] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}

    f1s = []
    for v in out.values ():
        val = v ["f1"]
        if val == val:
            f1s.append (val)

    if len (f1s) > 0:
        macro_f1 = sum (f1s) / len (f1s)
    else:
        macro_f1 = float ("nan")
    out ["macro_f1"] = {"f1": macro_f1}

    correct_count = 0
    for i in range (len (y_true)):
        if y_true [i] == y_pred [i]:
            correct_count = correct_count + 1

    if len (y_true) > 0:
        accuracy = correct_count / len (y_true)
    else:
        accuracy = float ("nan")
    out ["accuracy"] = {"acc": accuracy}

    return out

# build confusion matrix between true and predicted labels
def confusion (y_true, y_pred, labels):
    matrix = {}
    
    for t in labels:
        row = {}
        for p in labels:
            count = 0
            for i in range (len (y_true)):
                if y_true [i] == t and y_pred [i] == p:
                    count = count + 1
            row [p] = count
        matrix [t] = row
    
    return matrix
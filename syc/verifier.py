from urllib.parse import urlparse

# clean text by removing punctuation and extra spaces
def _norm (s):
    chars = []
    
    for ch in s.lower ():
        if ch.isalnum () or ch == " ":
            chars.append (ch)
    joined = "".join (chars)
    return " ".join (joined.split ())

# normalize url to compare easily
def _url_key (u):
    p = urlparse (u)
    domain = p.netloc.lower ().removeprefix ("www.")
    path = p.path.rstrip ("/")
    return (domain + path).lower ()

# extract quoted text from message
def _extract_quotes (message):
    quotes = []
    quote_pairs = [('"', '"'), ("\u201c", "\u201d")]
    
    for open_mark, close_mark in quote_pairs:
        pos = 0
        
        while True:
            start = message.find (open_mark, pos)
            if start < 0:
                break
            end = message.find (close_mark, start + 1)
            if end < 0:
                break
            q = message [start + 1 : end]
            if len (q) >= 8 and len (q) <= 400 and len (q.split ()) >= 4:
                quotes.append (q)
            pos = end + 1
    
    return quotes

# extract web urls from message
def _extract_urls (message):
    urls = []
    strip_chars = "()[]\"'<>.,;:"
    
    for word in message.split ():
        w = word.strip (strip_chars)
        if w.startswith ("http://") or w.startswith ("https://"):
            urls.append (w)
    return urls

# extract bracketed citation reference numbers
def _extract_refs (message):
    refs = []
    pos = 0
    
    while True:
        start = message.find ("[", pos)
        if start < 0:
            break
        end = message.find ("]", start + 1)
        if end < 0:
            break
        inside = message [start + 1 : end].strip ()
        if inside.isdigit () and len (inside) <= 2:
            refs.append (int (inside))
        pos = end + 1
    
    return refs

# check if quotes and urls exist in retrieved search results
def verify_message (message, results):
    quotes = _extract_quotes (message)
    urls = _extract_urls (message)
    refs = _extract_refs (message)

    corpus = []
    i = 0
    
    for r in results:
        text_norm = _norm (r.title + " " + r.snippet)
        corpus.append ((i + 1, text_norm))
        i = i + 1

    result_urls = set ()
    
    for r in results:
        result_urls.add (_url_key (r.url))

    cits = []
    verified = False
    
    for q in quotes:
        nq = _norm (q)
        hit = None
        for idx, text in corpus:
            if nq and nq in text:
                hit = idx
                break

        if hit and (not refs or hit in refs):
            verified = True
            cits.append ({"type": "quote", "text": q [ : 120], "status": "VERIFIED_QUOTE", "result": hit})
        elif hit:
            cits.append ({"type": "quote", "text": q [ : 120], "status": "QUOTE_FOUND_BUT_WRONG_REFERENCE", "result": hit})
        else:
            cits.append ({"type": "quote", "text": q [ : 120], "status": "QUOTE_NOT_FOUND_IN_RETRIEVED_SOURCES"})
    
    for u in urls:
        ok = _url_key (u) in result_urls
        if ok:
            status = "URL_IN_RETRIEVED_SOURCES"
        else:
            status = "URL_NOT_IN_RETRIEVED_SOURCES"
        cits.append ({"type": "url", "text": u [ : 120], "status": status})
    
    if not cits:
        report = "No verifiable citation found: the message contains no quoted source text and no URL."
    else:
        report_lines = []
        for c in cits:
            report_lines.append (f"- {c ['type']} \"{c ['text']}\" -> {c ['status']}")
        report = "\n".join (report_lines)

    if not results:
        report = report + "\n(No external sources were retrieved for this exchange.)"

    return {"verified": verified, "citations": cits, "report": report}
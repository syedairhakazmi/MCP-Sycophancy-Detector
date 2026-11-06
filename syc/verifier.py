import unicodedata
from urllib.parse import urlparse

MIN_FRAGMENT = 12  
MIN_QUOTE = 20      

def find_quotes (text):
    results = []
    start = 0

    while start < len (text):
        open_pos = -1
        close_pos = -1

        for quote in ['"', '“']:
            pos = text.find (quote, start)
            if pos != -1 and (open_pos == -1 or pos < open_pos):
                open_pos = pos
        
        if open_pos == -1:
            break
        
        if text [open_pos] == '"':
            closing_quote = '"'
        else:
            closing_quote = '”'

        close_pos = text.find (closing_quote, open_pos + 1)

        if close_pos == -1:
            break

        content = text [open_pos + 1 : close_pos]
        
        if 8 <= len (content) <= 400:
            results.append (content)
        
        start = close_pos + 1

    return results

def find_urls (text):
    results = []

    for word in text.split ():
        word = word.strip (')]}"\'')
        if word.startswith ("http://") or word.startswith ("https://"):
            results.append (word)
    
    return results

def find_references (text):
    results = []
    i = 0

    while i < len (text):
        if text [i] == "[":
            j = i + 1
            number = ""

            while j < len (text) and text [j].isdigit (): 
                number += text [j]
                j += 1

            if number and j < len (text) and text [j] == "]":
                if len (number) <= 2:
                    results.append (int (number))
                i = j   
        i += 1
    
    return results

def find_ellipses (text):
    results = []
    i = 0

    while i < len (text):
        if text [i] == '_':
            results.append ('_')
            i += 1

        elif text [i] == '.':
            count = 0

            while i < len (text) and text [i] == '.':
                count += 1
                i += 1

            if count >= 3:
                results.append ('.' * count) 
        
        elif text [i : i + 5] == '[...]':
            results.append ('[...]')
            i += 5
       
        else:
            i += 1

    return results    

def squash (s):
    # convert to lowercase and keep only letters and numbers
    s = unicodedata.normalize ("NFKD", s).casefold ()
    result = ""

    for ch in s:
        if ch.isalnum ():
            result += ch

    return result

def url_key(u):
    p = urlparse (u)

    domain = p.netloc.lower ()

    if domain.startswith ("www."):
        domain = domain [4 : ]

    path = p.path.rstrip ("/")

    return (domain + path).lower ()

def find_quote(quote, corpus):
    # find the first result containing all quote fragments in order
    frags = []
    start = 0
    i = 0

    while i < len (quote):
        if quote [i] == '…':
            frags.append (squash (quote [start : i]))
            i += 1
            start = i

        elif quote [i] == '.':
            count = 0

            while i < len (quote) and quote [i] == '.':
                count += 1
                i += 1

            if count >= 3:
                frags.append (squash (quote [start : i - count]))
                start = i

        elif quote [i : i + 5] == '[...]':
            frags.append (squash (quote [start : i]))
            i += 5
            start = i

        else:
            i += 1

    frags.append (squash (quote [start : ]))

    # keep fragments that are long enough
    valid_frags = []

    for f in frags:
        if len (f) >= MIN_FRAGMENT:
            valid_frags.append (f)

    # check if enough quote text is available
    total_length = 0

    for f in valid_frags:
        total_length += len (f)

    if not valid_frags or total_length < MIN_QUOTE:
        return None

    # search for fragments in each result
    for idx, text in corpus:
        pos = 0
        ok = True

        for f in valid_frags:
            p = text.find (f, pos)

            if p < 0:
                ok = False
                break

            pos = p + len (f)

        if ok:
            return idx

    return None

def verify_message (message, results):
    quotes = []

    for q in find_quotes (message):
        if len (q.split ()) >= 4:
            quotes.append (q)

    urls = find_urls (message)
    refs = find_references (message)

    corpus = []

    i = 0
    for r in results:
        text = squash (r.title + " " + r.snippet)
        corpus.append ((i + 1, text))
        i += 1

    result_urls = set ()

    for r in results:
        result_urls.add (url_key (r.url))

    cits = []
    verified = False

    for q in quotes:
        hit = find_quote (q, corpus)

        if hit and (not refs or hit in refs):
            verified = True

            cits.append ({
                "type": "quote",
                "text": q [ : 120],
                "status": "VERIFIED_QUOTE",
                "result": hit
            })

        elif hit:
            cits.append ({
                "type": "quote",
                "text": q [ : 120],
                "status": "QUOTE_FOUND_BUT_WRONG_REFERENCE",
                "result": hit
            })

        else:
            cits.append ({
                "type": "quote",
                "text": q [ : 120],
                "status": "QUOTE_NOT_FOUND_IN_RETRIEVED_SOURCES"
            })

    for u in urls:
        ok = url_key (u) in result_urls

        if ok:
            status = "URL_IN_RETRIEVED_SOURCES"
        else:
            status = "URL_NOT_IN_RETRIEVED_SOURCES"

        cits.append ({
            "type": "url",
            "text": u [ : 120],
            "status": status
        })

    if not cits:
        report = "No verifiable citation found: the message contains no quoted source text and no URL."

    else:
        report_lines = []

        for c in cits:
            line = "- " + c ["type"] + ' "' + c ["text"] + '" -> ' + c ["status"]
            report_lines.append (line)

        report = "\n".join (report_lines)

    if not results:
        report += "\n(No external sources were retrieved for this exchange.)"

    return {
        "verified": verified,
        "citations": cits,
        "report": report
    }
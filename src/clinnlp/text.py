"""Tokenisation and sentence splitting with character offsets."""
import re

TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]|_")
_ABBREV = {"dr", "mr", "mrs", "ms", "st", "vs", "e.g", "i.e", "approx", "no", "pt", "hx", "fx"}


def tokenize(text, offset=0):
    return [(m.group(), m.start() + offset, m.end() + offset) for m in TOKEN_RE.finditer(text)]


def split_sentences(text):
    """Return [(start, end)] spans. Splits on . ! ? and blank lines / list newlines, guarding common abbreviations."""
    spans, start, i, n = [], 0, 0, len(text)
    while i < n:
        c = text[i]
        brk = False
        if c in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?)\"'":
                j += 1
            k = j
            while k < n and text[k] in " \t":
                k += 1
            word = re.search(r"([A-Za-z.]+)$", text[max(0, i - 6):i])
            is_abbrev = bool(word) and c == "." and word.group(1).lower().strip(".") in _ABBREV
            is_decimal = c == "." and i > 0 and text[i - 1].isdigit() and i + 1 < n and text[i + 1].isdigit()
            if not is_abbrev and not is_decimal and (k >= n or text[k].isupper() or text[k].isdigit() or text[k] == "\n"):
                brk, i = True, j
        elif c == "\n":
            brk = True
            i += 1
            # a newline only ends a sentence when it is followed by another newline or a heading-like line
            nxt = text[i:i + 60].split("\n", 1)[0]
            if not (text[i:i + 1] == "\n" or re.match(r"\s*[A-Z][A-Za-z /&]{2,40}:", nxt) or re.match(r"\s*[-*\d]+[.)]?\s", nxt)):
                brk = False
        if brk:
            if text[start:i].strip():
                spans.append((start, i))
            start = i
        else:
            i += 1
    if text[start:].strip():
        spans.append((start, n))
    out = []
    for s, e in spans:
        seg = text[s:e]
        s2 = s + (len(seg) - len(seg.lstrip()))
        e2 = e - (len(seg) - len(seg.rstrip()))
        if e2 > s2:
            out.append((s2, e2))
    return out

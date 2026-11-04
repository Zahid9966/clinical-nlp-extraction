"""ConText/NegEx-style rule engine for assertion attributes of an already-located entity.

For each entity we return three attributes:
  assertion   : present | absent | possible | hypothetical
  temporality : current | historical
  experiencer : patient | family
plus the triggers that fired (for explainability).

Algorithm (Chapman et al. 2007 / Harkema et al. 2009, re-implemented from the published description):
trigger phrases open a scope (forward, backward or bidirectional); scopes are closed by terminators
(e.g. "but", "however") and by the sentence boundary; pseudo-triggers ("no change", "not only") suppress a
trigger; section headings (Family History, Past Medical History) set a default for the whole section.
"""
import re
from dataclasses import dataclass

from .text import split_sentences, tokenize

# (phrase, category, direction)  category: NEG | POSS | HYPO | HIST | FAM ; direction: F(orward) B(ackward) U(bidirectional)
TRIGGERS = [
    # negation
    ("no", "NEG", "F"), ("denies", "NEG", "F"), ("denied", "NEG", "F"), ("deny", "NEG", "F"), ("without", "NEG", "F"),
    ("negative for", "NEG", "F"), ("no evidence of", "NEG", "F"), ("no sign of", "NEG", "F"), ("no signs of", "NEG", "F"),
    ("no history of", "NEG", "F"), ("no complaints of", "NEG", "F"), ("free of", "NEG", "F"), ("absence of", "NEG", "F"),
    ("not", "NEG", "F"), ("never had", "NEG", "F"), ("never", "NEG", "F"), ("fails to reveal", "NEG", "F"),
    ("no family history of", "NEG", "F"), ("no suspicion of", "NEG", "F"), ("no findings of", "NEG", "F"),
    ("rules out", "NEG", "F"), ("ruled out for", "NEG", "F"), ("neg for", "NEG", "F"), ("w/o", "NEG", "F"),
    ("negative", "NEG", "B"), ("was ruled out", "NEG", "B"), ("is ruled out", "NEG", "B"), ("ruled out", "NEG", "B"),
    ("is absent", "NEG", "B"), ("was absent", "NEG", "B"), ("not seen", "NEG", "B"), ("is unlikely", "NEG", "B"),
    ("unlikely", "NEG", "B"), ("was excluded", "NEG", "B"), ("excluded", "NEG", "B"), ("not present", "NEG", "B"),
    ("not found", "NEG", "B"), ("is not present", "NEG", "B"), ("was not found", "NEG", "B"), ("has resolved", "NEG", "B"),
    # uncertainty
    ("possible", "POSS", "F"), ("probable", "POSS", "F"), ("suspected", "POSS", "F"), ("suspicious for", "POSS", "F"),
    ("likely", "POSS", "F"), ("concern for", "POSS", "F"), ("concerning for", "POSS", "F"), ("cannot rule out", "POSS", "F"),
    ("can not rule out", "POSS", "F"), ("cannot exclude", "POSS", "F"), ("differential includes", "POSS", "F"),
    ("differential diagnosis includes", "POSS", "F"), ("suggestive of", "POSS", "F"), ("consistent with", "POSS", "F"),
    ("may have", "POSS", "F"), ("might have", "POSS", "F"), ("question of", "POSS", "F"), ("query", "POSS", "F"),
    ("r/o", "POSS", "F"), ("rule out", "POSS", "F"), ("evaluate for", "POSS", "F"), ("versus", "POSS", "U"), ("vs", "POSS", "U"),
    ("suggests", "POSS", "F"), ("appears to be", "POSS", "F"), ("is probable", "POSS", "B"), ("is possible", "POSS", "B"),
    ("is suspected", "POSS", "B"), ("is likely", "POSS", "B"), ("cannot be excluded", "POSS", "B"), ("is not excluded", "POSS", "B"),
    ("?", "POSS", "F"),
    # hypothetical / future / prophylaxis
    ("if", "HYPO", "F"), ("should", "HYPO", "F"), ("in case of", "HYPO", "F"), ("return if", "HYPO", "F"),
    ("monitor for", "HYPO", "F"), ("watch for", "HYPO", "F"), ("signs of", "HYPO", "F"), ("risk of", "HYPO", "F"),
    ("risk for", "HYPO", "F"), ("prevention of", "HYPO", "F"), ("to prevent", "HYPO", "F"), ("screening for", "HYPO", "F"),
    ("at risk for", "HYPO", "F"), ("precautions for", "HYPO", "F"), ("call if", "HYPO", "F"), ("in the event of", "HYPO", "F"),
    ("develops", "HYPO", "B"), ("develop", "HYPO", "B"), ("occurs", "HYPO", "B"), ("recurs", "HYPO", "B"),
    # historical
    ("history of", "HIST", "F"), ("h/o", "HIST", "F"), ("hx of", "HIST", "F"), ("previously", "HIST", "F"), ("prior", "HIST", "F"),
    ("past", "HIST", "F"), ("remote", "HIST", "F"), ("s/p", "HIST", "F"), ("status post", "HIST", "F"), ("previous", "HIST", "F"),
    ("treated for", "HIST", "F"), ("history significant for", "HIST", "F"), ("years ago", "HIST", "B"), ("as a child", "HIST", "B"),
    ("resolved", "HIST", "B"), ("in remission", "HIST", "B"), ("last year", "HIST", "B"), ("in childhood", "HIST", "B"),
    # experiencer
    ("mother", "FAM", "F"), ("father", "FAM", "F"), ("brother", "FAM", "F"), ("sister", "FAM", "F"), ("sibling", "FAM", "F"),
    ("siblings", "FAM", "F"), ("grandmother", "FAM", "F"), ("grandfather", "FAM", "F"), ("aunt", "FAM", "F"), ("uncle", "FAM", "F"),
    ("son", "FAM", "F"), ("daughter", "FAM", "F"), ("cousin", "FAM", "F"), ("family history", "FAM", "F"), ("fh", "FAM", "F"),
    ("family history of", "FAM", "F"), ("family history significant for", "FAM", "F"), ("maternal", "FAM", "F"), ("paternal", "FAM", "F"),
    ("his mother", "FAM", "F"), ("her mother", "FAM", "F"), ("his father", "FAM", "F"), ("her father", "FAM", "F"),
    ("runs in the family", "FAM", "B"), ("no family history of", "FAM", "F"),
]
# phrases that look like triggers but are not
PSEUDO = [
    "no change", "no changes", "no increase", "no decrease", "no further", "not only", "not certain if", "not certain whether",
    "no doubt", "without difficulty", "not cause", "not necessarily", "gram negative", "no longer", "not exclude", "not rule out",
    "no new", "no worsening", "no improvement", "not limited to", "not sure", "without further", "not changed", "no interval change",
    "if not", "history and physical", "history of present illness", "past medical history", "no known",  # 'no known' handled below
]
PSEUDO = [p for p in PSEUDO if p != "no known"]
TERMINATORS = {"but", "however", "although", "though", "except", "aside", "apart", "which", "since", "because", "whereas",
               "yet", "while", "who", "that", "presents", "reports", "has", "have", "had", "complains", "complained", "admitted",
               "notes", "demonstrates", "shows", "showed", "developed"}
HARD_TERM = {"but", "however", "although", "though", "except", "whereas", "yet", "while", "which", "since", "because"}
SECTIONS = [
    (re.compile(r"^\s*(family history|fh|family hx)\b[:\s]", re.I), {"experiencer": "family"}),
    (re.compile(r"^\s*(past medical history|pmh|past history|medical history|history of past illness|pmhx)\b[:\s]", re.I), {"temporality": "historical"}),
    (re.compile(r"^\s*(past surgical history|psh)\b[:\s]", re.I), {"temporality": "historical"}),
    (re.compile(r"^\s*(plan|recommendations|discharge instructions|return precautions)\b[:\s]", re.I), {}),
]


@dataclass
class Assertion:
    assertion: str = "present"
    temporality: str = "current"
    experiencer: str = "patient"
    triggers: tuple = ()

    def as_dict(self):
        return {"assertion": self.assertion, "temporality": self.temporality, "experiencer": self.experiencer,
                "triggers": list(self.triggers)}


def _phrase_table():
    tbl = {}
    for p, cat, d in TRIGGERS:
        tbl.setdefault(tuple(p.split()) if p == "?" else tuple(re.findall(r"[a-z/]+|\?", p)), []).append((cat, d))
    return tbl


_TABLE = _phrase_table()
_PSEUDO = {tuple(p.split()) for p in PSEUDO}
_MAXN = max(len(k) for k in list(_TABLE) + list(_PSEUDO))


def _find_triggers(words):
    """Return list of (start_idx, end_idx_exclusive, category, direction, phrase). Longest match wins; pseudo phrases consume."""
    out, i, n = [], 0, len(words)
    while i < n:
        matched = False
        for L in range(min(_MAXN, n - i), 0, -1):
            seg = tuple(words[i:i + L])
            if seg in _PSEUDO:
                i += L
                matched = True
                break
            if seg in _TABLE:
                for cat, d in _TABLE[seg]:
                    out.append((i, i + L, cat, d, " ".join(seg)))
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return out


def section_defaults(text, pos):
    """Section defaults from the nearest preceding heading line."""
    line_start = text.rfind("\n", 0, pos) + 1
    while True:
        line = text[line_start:text.find("\n", line_start) if text.find("\n", line_start) != -1 else len(text)]
        for rx, d in SECTIONS:
            if rx.match(line):
                return d
        if re.match(r"^\s*[A-Z][A-Za-z /&]{2,40}:", line) and line_start != text.rfind("\n", 0, pos) + 1:
            return {}
        if line_start == 0:
            return {}
        prev = text.rfind("\n", 0, line_start - 1) + 1
        # headings are on their own line; stop at a blank line
        if not text[prev:line_start].strip():
            return {}
        line_start = prev
        # only look back a few lines
        if text.count("\n", line_start, pos) > 6:
            return {}


def classify(text, entities):
    """entities: list of (start, end). Returns list[Assertion] aligned with input."""
    sents = split_sentences(text)
    results = [None] * len(entities)
    for idx, (es, ee) in enumerate(entities):
        sent = next(((s, e) for s, e in sents if s <= es < e), (0, len(text)))
        s0, e0 = sent
        toks = tokenize(text[s0:e0], s0)
        words = [t[0].lower() for t in toks]
        # entity token index range
        ti = [k for k, (_, s, e) in enumerate(toks) if s < ee and e > es]
        if not ti:
            results[idx] = Assertion()
            continue
        a, b = ti[0], ti[-1]
        trig = _find_triggers(words)
        # terminator positions
        terms = [k for k, w in enumerate(words) if w in TERMINATORS or w in {";", ":"} and False]
        fired = []
        votes = {"NEG": False, "POSS": False, "HYPO": False, "HIST": False, "FAM": False}
        for (ts, te, cat, d, phrase) in trig:
            if ts <= b and te > a:  # trigger overlaps entity
                continue
            if d in ("F", "U") and te <= a:
                lo, hi = te, a
                if not _blocked(words, lo, hi, cat):
                    fired.append((cat, phrase)); votes[cat] = True
            if d in ("B", "U") and ts > b:
                lo, hi = b + 1, ts
                if not _blocked(words, lo, hi, cat):
                    fired.append((cat, phrase)); votes[cat] = True
        default = section_defaults(text, es)
        exp = "family" if votes["FAM"] or default.get("experiencer") == "family" else "patient"
        temp = "historical" if votes["HIST"] or default.get("temporality") == "historical" else "current"
        if votes["NEG"]:
            asr = "absent"
        elif votes["HYPO"]:
            asr = "hypothetical"
        elif votes["POSS"]:
            asr = "possible"
        else:
            asr = "present"
        results[idx] = Assertion(asr, temp, exp, tuple(f"{c}:{p}" for c, p in dict.fromkeys(fired)))
    return results


def _blocked(words, lo, hi, cat):
    """Is the scope between trigger and entity cut by a terminator (or, for FAM/HIST, a sentence-internal hard break)?"""
    between = words[lo:hi]
    if len(between) > 12 and cat != "FAM":
        return True
    for w in between:
        if w in HARD_TERM:
            return True
        if cat in ("NEG", "POSS", "HYPO") and w in TERMINATORS:
            return True
        if w in {";"}:
            return True
    return False

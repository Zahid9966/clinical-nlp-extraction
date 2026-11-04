"""PHI de-identification: regular expressions for structured identifiers + a CRF for names and locations.

Categories (a subset of HIPAA Safe Harbor identifiers): NAME, LOCATION, DATE, PHONE, ID, EMAIL, AGE (>= 90).
"""
import re
from pathlib import Path

import pycrfsuite

from .ner import shape
from .text import split_sentences, tokenize

MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
RULES = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)")),
    ("DATE", re.compile(rf"\b(?:{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}\s+{MONTH}\.?,?\s+\d{{4}}|{MONTH}\s+\d{{4}})\b")),
    ("DATE", re.compile(r"(?<![\d/])(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))(?![\d/])")),
    ("ID", re.compile(r"\b(?:MRN|ID|Acct|Account|SSN)\s*[:#]?\s*[\dA-Z][\dA-Z-]{4,}\b|\b\d{2}-\d{4}-\d{3}\b|\b\d{3}-\d{2}-\d{4}\b")),
    ("AGE", re.compile(r"\b(?:9\d|1[0-4]\d)(?=-year-old|\s*(?:years? old|y/?o)\b)(?:-year-old|\s*(?:years? old|y/?o))?")),
]
TITLE_CUES = {"dr", "mr", "mrs", "ms", "miss", "prof", "nurse", "md"}


def regex_spans(text):
    spans = []
    for typ, rx in RULES:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), typ))
    return _resolve(spans)


TITLE_NAME = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Miss|Prof|Nurse)\.?\s+([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)?)")
_COMMON_CAPS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "english", "spanish", "american", "african", "asian",
                "caucasian", "hispanic", "european", "chinese", "japanese", "indian", "medicare", "medicaid",
                "admitted", "discharged", "temp", "patient", "name", "phone", "history", "present", "illness", "systems", "review", "assessment",
                "impression", "plan", "recommendations", "family", "past", "medical", "social", "surgical", "medications", "allergies"}
_DISEASE_NOUNS = {"syndrome", "disease", "disorder", "deficiency", "anemia", "anaemia", "dystrophy", "type", "carcinoma", "lymphoma", "atrophy", "ataxia"}


def title_name_spans(text):
    return [(m.start(1), m.end(1), "NAME") for m in TITLE_NAME.finditer(text)]


def aggressive_spans(text, existing, protect=()):
    """High-recall backstop: capitalised, non-sentence-initial words that are not medical vocabulary are treated as possible names.
    Trades precision for recall (use when missing PHI is costlier than over-redaction)."""
    voc, out = med_vocab(), []
    for s, e in split_sentences(text):
        toks = tokenize(text[s:e], s)
        for i, (w, a, b) in enumerate(toks):
            if i == 0 or not (w.istitle() and w.isalpha() and len(w) >= 3):
                continue
            if w.lower() in voc or w.lower() in _COMMON_CAPS or toks[i - 1][0] in ".!?:":
                continue
            if any(a < x[1] and x[0] < b for x in list(existing) + list(protect)):
                continue
            if i + 1 < len(toks) and toks[i + 1][0].lower() in _DISEASE_NOUNS:
                continue
            ls = text.rfind("\n", 0, a) + 1
            le = text.find("\n", a); le = len(text) if le < 0 else le
            if text[ls:le].rstrip().endswith(":") and le - ls < 50:   # heading line
                continue
            out.append((a, b, "NAME"))
    return out


def _resolve(spans):
    """Drop spans overlapped by a longer span (ties: earlier rule wins)."""
    spans = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    keep = []
    for s in spans:
        if not any(s[0] < k[1] and k[0] < s[1] for k in keep):
            keep.append(s)
    return sorted(keep)


_VOCAB = None


def med_vocab():
    """Lower-cased word types seen in MeSH disease terms: capitalised words outside it are likelier to be names/places."""
    global _VOCAB
    if _VOCAB is None:
        import json
        from .ner import LEXICON
        v = set()
        try:
            for rec in json.loads(Path(LEXICON).read_text()).values():
                for t in rec["terms"]:
                    v.update(w.lower() for w, _, _ in tokenize(t))
        except FileNotFoundError:
            pass
        _VOCAB = v
    return _VOCAB


def featurize(words):
    low = [w.lower() for w in words]
    voc = med_vocab()
    out = []
    for i, w in enumerate(words):
        f = {"bias": 1.0, "w": low[i], "shape": shape(w), "title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(), "first": i == 0,
             "s3": low[i][-3:], "p3": low[i][:3], "len": min(len(w), 10)}
        for d in (-3, -2, -1, 1, 2, 3):
            j = i + d
            if 0 <= j < len(words):
                f[f"{d}:w"] = low[j]
                f[f"{d}:title"] = words[j].istitle()
                if abs(d) == 1:
                    f[f"{d}:shape"] = shape(words[j])
            else:
                f[f"{d}:pad"] = True
        f["in_med_vocab"] = low[i] in voc
        f["cap_oov"] = words[i].istitle() and low[i] not in voc
        for d in (-1, 1):
            j = i + d
            if 0 <= j < len(words):
                f[f"{d}:cap_oov"] = words[j].istitle() and low[j] not in voc
        f["prev_title_cue"] = i > 0 and low[i - 1] in TITLE_CUES or (i > 1 and low[i - 1] == "." and low[i - 2] in TITLE_CUES)
        f["prev_title_cap_nonfirst"] = words[i].istitle() and i > 0 and words[i - 1] not in ".!?:\n"
        f["bi-1"] = (low[i - 1] if i else "<s>") + "|" + low[i]
        out.append(f)
    return out


def merge_adjacent(spans, text):
    """Merge neighbouring same-type NAME/LOCATION spans separated only by a space (first + last name)."""
    out = []
    for sp in sorted(spans):
        if out and out[-1][2] == sp[2] and sp[2] in ("NAME", "LOCATION") and text[out[-1][1]:sp[0]].strip() == "" and sp[0] - out[-1][1] <= 1:
            out[-1] = (out[-1][0], sp[1], sp[2])
        else:
            out.append(sp)
    return out


class Deidentifier:
    def __init__(self, model_path=None):
        self.tagger = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path):
        self.tagger = pycrfsuite.Tagger(); self.tagger.open(str(path)); return self

    def _sents(self, text):
        for s, e in split_sentences(text):
            toks = tokenize(text[s:e], s)
            if toks:
                yield toks

    def fit(self, notes, model_path, c1=0.1, c2=0.05, max_iter=120):
        tr = pycrfsuite.Trainer(verbose=False)
        for n in notes:
            phi = [p for p in n.phi if p[2] in ("NAME", "LOCATION")]
            for toks in self._sents(n.text):
                labels = ["O"] * len(toks)
                for s, e, t in phi:
                    first = True
                    for i, (_, a, b) in enumerate(toks):
                        if a >= s and b <= e:
                            labels[i] = ("B-" if first else "I-") + t; first = False
                tr.append(featurize([t[0] for t in toks]), labels)
        tr.set_params({"c1": c1, "c2": c2, "max_iterations": max_iter, "feature.possible_transitions": True})
        tr.train(str(model_path))
        return self.load(model_path)

    def crf_spans(self, text):
        if self.tagger is None:
            return []
        res = []
        for toks in self._sents(text):
            self.tagger.set(featurize([t[0] for t in toks]))
            tags = self.tagger.tag()
            i = 0
            while i < len(tags):
                if tags[i] != "O":
                    typ = tags[i][2:]
                    j = i
                    while j + 1 < len(tags) and tags[j + 1] == "I-" + typ:
                        j += 1
                    res.append((toks[i][1], toks[j][2], typ))
                    i = j + 1
                else:
                    i += 1
        return res

    def detect(self, text, use_crf=True, aggressive=False, protect=()):
        """Return sorted [(start, end, type)]. Layers: regex rules -> title+name rule -> CRF -> (optional) aggressive backstop."""
        spans = regex_spans(text)
        extra = title_name_spans(text) + (self.crf_spans(text) if use_crf else [])
        spans = _resolve(spans + [s for s in extra if not any(s[0] < r[1] and r[0] < s[1] for r in spans)])
        if aggressive:
            spans = _resolve(spans + aggressive_spans(text, spans, protect))
        return merge_adjacent(spans, text)

    @staticmethod
    def redact(text, spans, style="tag"):
        """style='tag' -> [NAME]; 'mask' -> same-length asterisks."""
        out, last = [], 0
        for s, e, t in sorted(spans):
            out.append(text[last:s])
            out.append(f"[{t}]" if style == "tag" else "*" * (e - s))
            last = e
        out.append(text[last:])
        return "".join(out)

"""CRF disease-mention recogniser (BIO tagging) trained on the NCBI Disease corpus."""
import json, re
from pathlib import Path

import pycrfsuite

from .ontology import norm
from .text import split_sentences, tokenize

LEXICON = Path(__file__).resolve().parents[2] / "data" / "raw" / "mesh_disease_lexicon.json"


def shape(w):
    s = re.sub(r"[A-Z]", "A", w)
    s = re.sub(r"[a-z]", "a", s)
    s = re.sub(r"\d", "0", s)
    return re.sub(r"(.)\1+", r"\1\1", s)


class Gazetteer:
    """Longest-match lookup of MeSH terms (token level) -> per-token B/I flags."""

    def __init__(self, lexicon=None):
        lex = lexicon if lexicon is not None else json.loads(LEXICON.read_text())
        self.terms, self.max_len = set(), 1
        for v in lex.values():
            for t in set(v["terms"]) | {v["name"]}:
                toks = tuple(w.lower() for w, _, _ in tokenize(t))
                if 0 < len(toks) <= 8 and not (len(toks) == 1 and len(toks[0]) < 4):
                    self.terms.add(toks)
                    self.max_len = max(self.max_len, len(toks))
        # single tokens that are suffix-like disease words give generalisation to unseen terms
        self.single = {t[0] for t in self.terms if len(t) == 1}

    def tag(self, words):
        low = [w.lower() for w in words]
        flags = ["O"] * len(words)
        i = 0
        while i < len(words):
            hit = 0
            for L in range(min(self.max_len, len(words) - i), 0, -1):
                if tuple(low[i:i + L]) in self.terms:
                    hit = L
                    break
            if hit:
                flags[i] = "B"
                for j in range(i + 1, i + hit):
                    flags[j] = "I"
                i += hit
            else:
                i += 1
        return flags


def featurize(words, gaz):
    g = gaz.tag(words)
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        f = {"bias": 1.0, "w": lw, "shape": shape(w), "gaz": g[i], "title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(),
             "len": min(len(w), 12), "single_dx": lw in gaz.single}
        for n in (2, 3, 4, 5):
            f[f"p{n}"] = lw[:n]
            f[f"s{n}"] = lw[-n:]
        for d in (-3, -2, -1, 1, 2, 3):
            j = i + d
            if 0 <= j < len(words):
                f[f"{d}:w"] = words[j].lower()
                if abs(d) <= 2:
                    f[f"{d}:shape"] = shape(words[j])
                    f[f"{d}:gaz"] = g[j]
                if abs(d) == 1:
                    f[f"{d}:s3"] = words[j].lower()[-3:]
            else:
                f[f"{d}:pad"] = True
        if i > 0:
            f["bi-1"] = words[i - 1].lower() + "|" + lw
        if i + 1 < len(words):
            f["bi+1"] = lw + "|" + words[i + 1].lower()
        out.append(f)
    return out


def bio_labels(tokens, mentions):
    """Flat BIO labels; for overlapping gold mentions keep the longest."""
    keep, last_end = [], -1
    for m in sorted(mentions, key=lambda m: (m.start, -(m.end - m.start))):
        if m.start >= last_end:
            keep.append(m)
            last_end = m.end
    labels = ["O"] * len(tokens)
    for m in keep:
        first = True
        for i, (_, s, e) in enumerate(tokens):
            if s >= m.start and e <= m.end:
                labels[i] = "B" if first else "I"
                first = False
    return labels


class DiseaseNER:
    def __init__(self, gazetteer=None):
        self.gaz = gazetteer or Gazetteer()
        self.tagger = None

    def _sentences(self, text):
        for s, e in split_sentences(text):
            toks = tokenize(text[s:e], s)
            if toks:
                yield toks

    def fit(self, docs, model_path, c1=0.05, c2=0.05, max_iter=150):
        tr = pycrfsuite.Trainer(verbose=False)
        for d in docs:
            for toks in self._sentences(d.text):
                tr.append(featurize([t[0] for t in toks], self.gaz), bio_labels(toks, d.mentions))
        tr.set_params({"c1": c1, "c2": c2, "max_iterations": max_iter, "feature.possible_transitions": True})
        tr.train(str(model_path))
        self.load(model_path)
        return self

    def load(self, model_path):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(model_path))
        return self

    def predict(self, text):
        """Return [(start, end, text, marginal_confidence)]."""
        res = []
        for toks in self._sentences(text):
            words = [t[0] for t in toks]
            self.tagger.set(featurize(words, self.gaz))
            tags = self.tagger.tag()
            i = 0
            while i < len(tags):
                if tags[i] in ("B", "I"):
                    j = i
                    while j + 1 < len(tags) and tags[j + 1] == "I":
                        j += 1
                    conf = min(self.tagger.marginal(tags[k], k) for k in range(i, j + 1))
                    s, e = toks[i][1], toks[j][2]
                    res.append((s, e, text[s:e], float(conf)))
                    i = j + 1
                else:
                    i += 1
        return res

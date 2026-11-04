"""MeSH disease normalisation: exact/normalised dictionary lookup, then char n-gram TF-IDF nearest synonym."""
import json, re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

LEXICON = Path(__file__).resolve().parents[2] / "data" / "raw" / "mesh_disease_lexicon.json"
_STOP = {"of", "the", "a", "an", "and", "in", "with", "due", "to"}


def norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = []
    for t in s.split():
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t.endswith("is") and not t.endswith("us"):
            t = t[:-1]
        toks.append(t)
    return " ".join(toks)


def key(s):
    """Order-insensitive normalised key ('cancer of the breast' ~ 'breast cancer')."""
    return " ".join(sorted(norm(s).split()))


class MeshNormalizer:
    def __init__(self, lexicon=None, extra_synonyms=None, threshold=0.62):
        lex = lexicon if lexicon is not None else json.loads(Path(LEXICON).read_text())
        self.names = {ui: v["name"] for ui, v in lex.items()}
        self.threshold = threshold
        self.exact, self.keyed, self._pairs = {}, {}, []
        surf = []
        for ui, v in lex.items():
            for t in set(v["terms"]) | {v["name"]}:
                self._add(t, ui, surf)
        for text, ui in (extra_synonyms or []):
            if ui in self.names:
                self._add(text, ui, surf)
        self.surf = sorted(set(surf))
        self._owner = {}
        for t, ui in self._pairs:
            self._owner.setdefault(norm(t), ui)
        self._terms = sorted(self._owner)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
        self.mat = self.vec.fit_transform(self._terms)

    def _add(self, text, ui, surf):
        self._pairs.append((text, ui))
        self.exact.setdefault(text.lower(), ui)
        self.keyed.setdefault(key(text), ui)
        surf.append(text)

    def lookup(self, mention):
        """Return (mesh_id | None, preferred_name | None, score, method)."""
        m = mention.strip()
        if m.lower() in self.exact:
            ui = self.exact[m.lower()]
            return ui, self.names[ui], 1.0, "exact"
        k = key(m)
        if k and k in self.keyed:
            ui = self.keyed[k]
            return ui, self.names[ui], 0.95, "normalized"
        if not k:
            return None, None, 0.0, "nil"
        sims = (self.mat @ self.vec.transform([norm(m)]).T).toarray().ravel()
        i = int(sims.argmax())
        if sims[i] >= self.threshold:
            ui = self._owner[self._terms[i]]
            return ui, self.names[ui], float(sims[i]), "fuzzy"
        return None, None, float(sims[i]), "nil"

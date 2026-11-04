"""Learned assertion classifier: logistic regression over local context words + ConText rule outputs (hybrid)."""
import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from . import context
from .text import split_sentences, tokenize

HEADS = ("assertion", "temporality", "experiencer")


def entity_features(text, ents, rules=None, use_rules=True):
    """ents: [(start, end)] -> list of feature dicts. Window words are limited to the entity's sentence."""
    rules = rules if rules is not None else context.classify(text, ents)
    sents = split_sentences(text)
    feats = []
    for (es, ee), r in zip(ents, rules):
        s0, e0 = next(((s, e) for s, e in sents if s <= es < e), (0, len(text)))
        toks = tokenize(text[s0:e0], s0)
        words = [t[0].lower() for t in toks]
        idx = [k for k, (_, a, b) in enumerate(toks) if a < ee and b > es]
        f = {"bias": 1.0}
        if idx:
            a, b = idx[0], idx[-1]
            for d in range(1, 9):
                if a - d >= 0:
                    w = words[a - d]
                    f[f"L{min(d, 4)}:{w}"] = 1.0
                    f[f"Lbag:{w}"] = 1.0 if d <= 6 else 0.5
            for d in range(1, 5):
                if b + d < len(words):
                    f[f"R{min(d, 3)}:{words[b + d]}"] = 1.0
                    f[f"Rbag:{words[b + d]}"] = 1.0
            if a >= 2:
                f[f"L2g:{words[a - 2]}_{words[a - 1]}"] = 1.0
            if a >= 3:
                f[f"L3g:{words[a - 3]}_{words[a - 2]}_{words[a - 1]}"] = 1.0
            f["sent_pos"] = a / max(len(words), 1)
        d = context.section_defaults(text, es)
        f[f"sec_exp:{d.get('experiencer', '-')}"] = 1.0
        f[f"sec_temp:{d.get('temporality', '-')}"] = 1.0
        line_start = text.rfind("\n", 0, es) + 1
        f[f"line_start:{text[line_start:line_start + 12].lower().strip()}"] = 1.0 if es - line_start < 14 else 0.0
        if use_rules:
            f[f"rule_a:{r.assertion}"] = 1.0
            f[f"rule_t:{r.temporality}"] = 1.0
            f[f"rule_e:{r.experiencer}"] = 1.0
            for t in r.triggers:
                f[f"trig:{t}"] = 1.0
        feats.append(f)
    return feats


class AssertionModel:
    def __init__(self, use_rules=True, C=3.0):
        self.use_rules, self.C = use_rules, C
        self.vec, self.clf = DictVectorizer(), {}

    def fit(self, notes):
        X, Y = [], {h: [] for h in HEADS}
        for n in notes:
            ents = [(e["start"], e["end"]) for e in n.entities]
            X += entity_features(n.text, ents, use_rules=self.use_rules)
            for h in HEADS:
                Y[h] += [e[h] for e in n.entities]
        M = self.vec.fit_transform(X)
        for h in HEADS:
            self.clf[h] = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced").fit(M, Y[h])
        return self

    def predict(self, text, ents):
        if not ents:
            return []
        M = self.vec.transform(entity_features(text, ents, use_rules=self.use_rules))
        P = {h: self.clf[h].predict(M) for h in HEADS}
        conf = {h: self.clf[h].predict_proba(M).max(axis=1) for h in HEADS}
        rules = context.classify(text, ents)
        return [context.Assertion(P["assertion"][i], P["temporality"][i], P["experiencer"][i], rules[i].triggers) for i in range(len(ents))], \
               [float(min(conf[h][i] for h in HEADS)) for i in range(len(ents))]

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        # pickle: only load model files you produced yourself with scripts/train.py
        return pickle.loads(Path(path).read_bytes())

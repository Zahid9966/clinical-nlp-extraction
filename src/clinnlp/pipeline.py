"""End-to-end pipeline: PHI detection -> disease NER -> assertion -> MeSH normalisation."""
import json
from pathlib import Path

from . import context
from .assertion import AssertionModel
from .deid import Deidentifier
from .ner import DiseaseNER
from .ontology import MeshNormalizer

MODELS = Path(__file__).resolve().parents[2] / "models"


class Pipeline:
    def __init__(self, models_dir=MODELS):
        d = Path(models_dir)
        self.ner = DiseaseNER().load(d / "ner.crf")
        self.deid = Deidentifier(d / "deid.crf")
        syn = json.loads((d / "train_synonyms.json").read_text()) if (d / "train_synonyms.json").exists() else []
        self.norm = MeshNormalizer(extra_synonyms=syn)
        self.assertion = AssertionModel.load(d / "assertion.pkl") if (d / "assertion.pkl").exists() else None

    def deidentify(self, text, aggressive=True, style="tag"):
        spans = self.deid.detect(text, aggressive=aggressive)
        return {"text": self.deid.redact(text, spans, style),
                "phi": [{"start": s, "end": e, "type": t, "text": text[s:e]} for s, e, t in spans]}

    def extract(self, text, aggressive_deid=True, assertion_mode="ml"):
        found = self.ner.predict(text)
        # disease mentions protect their tokens from the over-eager name backstop (e.g. "Pendred syndrome")
        phi = self.deid.detect(text, aggressive=aggressive_deid, protect=[(m[0], m[1]) for m in found])
        raw = [m for m in found
               if not any(s < m[1] and m[0] < e and t in ("NAME", "LOCATION") and (min(e, m[1]) - max(s, m[0])) >= 0.5 * (m[1] - m[0]) for s, e, t in phi)]
        ents = [(m[0], m[1]) for m in raw]
        if self.assertion and assertion_mode == "ml":
            ass, aconf = self.assertion.predict(text, ents)
        else:
            ass, aconf = context.classify(text, ents), [None] * len(ents)
        out = []
        for m, a, ac in zip(raw, ass, aconf):
            mesh, name, score, method = self.norm.lookup(m[2])
            out.append({"start": m[0], "end": m[1], "text": m[2], "label": "DISEASE", "ner_confidence": round(m[3], 4),
                        **{k: v for k, v in a.as_dict().items()}, "assertion_confidence": None if ac is None else round(ac, 4),
                        "mesh_id": mesh, "mesh_name": name, "mesh_score": round(score, 4), "mesh_method": method})
        return {"entities": out, "phi": [{"start": s, "end": e, "type": t} for s, e, t in phi],
                "deidentified_text": self.deid.redact(text, phi)}

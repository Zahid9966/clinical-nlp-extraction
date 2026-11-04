"""Train all models: disease CRF (NCBI train), de-id CRF + assertion model (synthetic train notes), save to models/."""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinnlp.assertion import AssertionModel
from clinnlp.corpus import load_ncbi
from clinnlp.deid import Deidentifier
from clinnlp.ner import DiseaseNER
from clinnlp.synth import Generator, disease_pool

M = ROOT / "models"
M.mkdir(exist_ok=True)
SEED_TRAIN = 1


def main():
    t = time.time()
    train = load_ncbi("train")
    DiseaseNER().fit(train, M / "ner.crf")
    print(f"NER trained on {len(train)} abstracts ({time.time() - t:.0f}s)")
    syn = sorted({(m.text, m.concept) for d in train for m in d.mentions if m.concept and "OMIM" not in m.concept and "|" not in m.concept})
    (M / "train_synonyms.json").write_text(json.dumps(syn))
    pool = disease_pool()
    notes = Generator("train", "train", seed=SEED_TRAIN, pool=pool).notes(1500)
    Deidentifier().fit(notes, M / "deid.crf", c1=0.0, c2=0.3)
    AssertionModel().fit(notes).save(M / "assertion.pkl")
    print(f"de-id + assertion trained on {len(notes)} synthetic notes ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

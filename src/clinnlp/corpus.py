"""Loader for the NCBI Disease corpus (PubTator format)."""
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass
class Mention:
    start: int
    end: int
    text: str
    label: str = "DISEASE"
    kind: str = ""          # SpecificDisease / DiseaseClass / Modifier / CompositeMention
    concept: str = ""       # e.g. "D003693" / "OMIM:123" / "D1|D2"


@dataclass
class Doc:
    pmid: str
    text: str
    mentions: list = field(default_factory=list)


def load_ncbi(split, path=None):
    f = Path(path) if path else DATA / {"train": "NCBItrainset_corpus.txt", "dev": "NCBIdevelopset_corpus.txt",
                                          "test": "NCBItestset_corpus.txt"}[split]
    docs, cur = {}, {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if "|t|" in line:
            pmid, _, t = line.split("|", 2)
            cur[pmid] = {"t": t, "a": "", "m": []}
        elif "|a|" in line:
            pmid, _, a = line.split("|", 2)
            cur[pmid]["a"] = a
        elif line.strip():
            p = line.split("\t")
            cur[p[0]]["m"].append(p)
    for pmid, d in cur.items():
        text = d["t"] + " " + d["a"]
        ms = []
        for p in d["m"]:
            s, e = int(p[1]), int(p[2])
            # a few mention strings in the TSV differ from the text only by quote/space normalisation; offsets are authoritative
            assert len(text[s:e]) == len(p[3]) or text[s:e].replace('"', " ").split() == p[3].replace('"', " ").split(), (pmid, text[s:e], p[3])
            ms.append(Mention(s, e, text[s:e], "DISEASE", p[4], p[5] if len(p) > 5 else ""))
        docs_ms = sorted(ms, key=lambda m: (m.start, m.end))
        docs[pmid] = Doc(pmid, text, docs_ms)
    return list(docs.values())

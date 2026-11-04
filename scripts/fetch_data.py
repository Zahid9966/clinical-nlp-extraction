"""Download the public corpora/vocabularies used by this project (no credentials needed).

* NCBI Disease corpus (NLM/NCBI, PubMed abstracts, public domain): train/dev/test with MeSH/OMIM concept IDs.
* MeSH 2026 descriptors (NLM, public domain): used to build the disease lexicon.
"""
import gzip, io, json, re, sys, urllib.request, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
NCBI = "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/{}.zip"
MESH = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"
SETS = ["NCBItrainset_corpus", "NCBIdevelopset_corpus", "NCBItestset_corpus"]


def get(url):
    print("GET", url, file=sys.stderr)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "clinnlp-fetch"}), timeout=120) as r:
        return r.read()


def build_mesh_lexicon(xml_bytes):
    """Keep descriptors under MeSH trees C (Diseases) and F03 (Mental Disorders) with all entry terms."""
    out = {}
    for _, el in ET.iterparse(io.BytesIO(xml_bytes), events=("end",)):
        if el.tag != "DescriptorRecord":
            continue
        trees = [t.text for t in el.findall("./TreeNumberList/TreeNumber")]
        if any(t.startswith("C") or t.startswith("F03") for t in trees):
            ui = el.findtext("DescriptorUI")
            name = el.findtext("./DescriptorName/String")
            terms = {t.text for t in el.findall(".//Term/String")}
            out[ui] = {"name": name, "terms": sorted(terms), "trees": trees}
        el.clear()
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        dest = RAW / f"{s}.txt"
        if not dest.exists():
            z = zipfile.ZipFile(io.BytesIO(get(NCBI.format(s))))
            dest.write_bytes(z.read(f"{s}.txt"))
    lex = RAW / "mesh_disease_lexicon.json"
    if not lex.exists():
        data = gzip.decompress(get(MESH))
        lex.write_text(json.dumps(build_mesh_lexicon(data)))
    print("done:", sorted(p.name for p in RAW.iterdir()))


if __name__ == "__main__":
    main()

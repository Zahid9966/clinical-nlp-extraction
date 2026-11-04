from conftest import needs_data

from clinnlp.corpus import load_ncbi
from clinnlp.ontology import MeshNormalizer, key, norm
from clinnlp.synth import Generator


def test_norm_and_key():
    assert key("cancer of the breast") == key("Breast cancers")
    assert norm("Diseases") == "disease"


@needs_data
def test_mesh_lookup_exact_and_fuzzy():
    nz = MeshNormalizer()
    assert nz.lookup("Diabetes Mellitus")[0] == "D003920"
    assert nz.lookup("breast cancer")[0] == "D001943"  # Breast Neoplasms (entry term)
    ui, _, score, method = nz.lookup("zzzz qqqq")
    assert ui is None and method == "nil"


@needs_data
def test_ncbi_loader_offsets():
    docs = load_ncbi("dev")
    assert len(docs) == 100
    for d in docs[:20]:
        for m in d.mentions:
            assert d.text[m.start:m.end] == m.text


@needs_data
def test_synthetic_notes_are_consistent_and_deterministic():
    a, b = Generator("train", "train", seed=5).note(), Generator("train", "train", seed=5).note()
    assert a.text == b.text
    for e in a.entities:
        assert a.text[e["start"]:e["end"]] == e["text"]
    for s, e, t in a.phi:
        assert a.text[s:e].strip()
    names_train = set(Generator("train", "train").first)
    names_test = set(Generator("test", "train").first)
    assert not names_train & names_test  # held-out first names

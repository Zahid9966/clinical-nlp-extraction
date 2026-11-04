import pytest
from conftest import needs_data, needs_models
from fastapi.testclient import TestClient

pytestmark = [needs_data, needs_models]

NOTE = ("Patient: John Whitaker  MRN: 7788991  phone (415) 555-0134\n\nHistory of Present Illness:\n"
        "Seen by Dr. Alice Moore on 03/14/2021. Denies diabetes mellitus. Possible pneumonia. Mother had breast cancer.\n")


@pytest.fixture(scope="module")
def client():
    from clinnlp.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and "Not a medical device" in r["disclaimer"]


def test_extract_end_to_end(client):
    r = client.post("/extract", json={"text": NOTE}).json()
    by = {e["text"].lower(): e for e in r["entities"]}
    assert by["diabetes mellitus"]["assertion"] == "absent" and by["diabetes mellitus"]["mesh_id"] == "D003920"
    assert by["pneumonia"]["assertion"] == "possible"
    assert by["breast cancer"]["experiencer"] == "family"
    kinds = {p["type"] for p in r["phi"]}
    assert {"DATE", "PHONE", "ID"} <= kinds
    assert "Whitaker" not in r["deidentified_text"] and "03/14/2021" not in r["deidentified_text"]


def test_deidentify_and_validation(client):
    r = client.post("/deidentify", json={"text": NOTE, "style": "mask"}).json()
    assert "415" not in r["text"]
    assert client.post("/extract", json={"text": ""}).status_code == 422
    assert client.post("/extract", json={"text": "x" * 60000}).status_code == 422
    assert client.post("/extract", json={"text": "a", "assertion_mode": "bogus"}).status_code == 422


def test_normalize(client):
    assert client.post("/normalize", json={"text": "type 2 diabetes mellitus"}).status_code in (200, 404)
    assert client.post("/normalize", json={"text": "Diabetes Mellitus"}).json()["mesh_id"] == "D003920"

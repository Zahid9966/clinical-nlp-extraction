import re

import pytest

from clinnlp.context import classify
from clinnlp.text import split_sentences, tokenize


def run(text, *terms):
    ents = [(m.start(), m.end()) for t in terms for m in re.finditer(re.escape(t), text)]
    return [a.as_dict() for a in classify(text, ents)]


def test_tokenize_offsets():
    text = "Dr. Smith saw him."
    for w, s, e in tokenize(text):
        assert text[s:e] == w


def test_sentence_split_abbrev_and_decimal():
    t = "Dr. Lee saw the patient. Temp 98.6 F. Plan follows."
    assert [t[s:e] for s, e in split_sentences(t)] == ["Dr. Lee saw the patient.", "Temp 98.6 F.", "Plan follows."]


@pytest.mark.parametrize("text,term,expected", [
    ("Patient denies chest pain.", "chest pain", "absent"),
    ("No evidence of pneumonia.", "pneumonia", "absent"),
    ("Pneumonia was ruled out.", "Pneumonia", "absent"),
    ("Possible pneumonia.", "pneumonia", "possible"),
    ("Cannot rule out pneumonia.", "pneumonia", "possible"),
    ("Return if sepsis develops.", "sepsis", "hypothetical"),
    ("Has asthma.", "asthma", "present"),
])
def test_assertion_basic(text, term, expected):
    assert run(text, term)[0]["assertion"] == expected


def test_pseudo_negation_does_not_negate():
    assert run("No change in diabetes.", "diabetes")[0]["assertion"] == "present"
    assert run("Not only asthma but also COPD.", "asthma")[0]["assertion"] == "present"


def test_scope_terminated_by_but():
    r = run("Denies fever but has pneumonia.", "fever", "pneumonia")
    assert [x["assertion"] for x in r] == ["absent", "present"]


def test_list_stays_in_scope():
    r = run("Denies asthma, diabetes or gout.", "asthma", "diabetes", "gout")
    assert all(x["assertion"] == "absent" for x in r)


def test_family_and_historical():
    assert run("Mother had breast cancer.", "breast cancer")[0]["experiencer"] == "family"
    assert run("History of tuberculosis.", "tuberculosis")[0]["temporality"] == "historical"


def test_section_defaults():
    t = "Family History:\nDiabetes. Asthma.\n\nPlan:\nRest."
    assert all(x["experiencer"] == "family" for x in run(t, "Diabetes", "Asthma"))
    t2 = "Past Medical History:\nHypertension.\n"
    assert run(t2, "Hypertension")[0]["temporality"] == "historical"


def test_no_family_history_is_absent_family():
    r = run("No family history of lupus.", "lupus")[0]
    assert r["assertion"] == "absent" and r["experiencer"] == "family"

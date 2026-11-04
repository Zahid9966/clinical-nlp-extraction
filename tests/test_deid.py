from clinnlp.deid import Deidentifier, regex_spans, title_name_spans


def types(text):
    return {(text[s:e], t) for s, e, t in regex_spans(text)}


def test_structured_phi_detected():
    t = "Call (555) 123-4567 or a.b@clinic.org. MRN: 1234567. Seen 03/14/2021 and March 3, 2019. She is 93-year-old."
    got = types(t)
    for expect in [("(555) 123-4567", "PHONE"), ("a.b@clinic.org", "EMAIL"), ("MRN: 1234567", "ID"), ("03/14/2021", "DATE"), ("March 3, 2019", "DATE"), ("93-year-old", "AGE")]:
        assert expect in got, (expect, got)


def test_clinical_numbers_are_not_phi():
    t = "BP 128/76, HR 82, creatinine 1.2 mg/dL, 54-year-old, seen in 2015, 500 mg twice daily."
    assert regex_spans(t) == []


def test_title_rule():
    t = "Referred to Dr. Alice Moore."
    assert [t[s:e] for s, e, _ in title_name_spans(t)] == ["Alice Moore"]


def test_redact_tag_and_mask():
    t = "Seen on 03/14/2021."
    spans = regex_spans(t)
    assert Deidentifier.redact(t, spans) == "Seen on [DATE]."
    assert Deidentifier.redact(t, spans, "mask") == "Seen on " + "*" * 10 + "."

# clinnlp — clinical NLP extraction

Disease mention extraction from clinical-style text with **assertion status** (negation, uncertainty, hypothetical, temporality, experiencer), **MeSH normalisation** and **PHI de-identification**, served by FastAPI.

> **Not for clinical use.** Research/portfolio prototype. Not a medical device, never validated on real clinical notes, and must not be used with real patient data or for any clinical decision.

## What is real and what is synthetic

| Component | Trained / evaluated on | Real? |
|---|---|---|
| Disease NER (CRF) | NCBI Disease corpus, PubMed abstracts (train 592 / test 100 docs) | **Real text**, but biomedical literature, not clinical notes |
| MeSH normalisation | MeSH 2026 disease vocabulary; NCBI Disease gold IDs | **Real** |
| Assertion (rules + ML) | Template-generated notes | **Synthetic** |
| De-identification | Template-generated notes with randomly generated PHI | **Synthetic** |
| End-to-end note numbers | Synthetic notes | **Synthetic** |

No MIMIC, i2b2/n2c2 or other credentialed data is used, so there is no real annotated clinical-note data here. Assertion and de-id results are therefore indicators of component behaviour, **not** estimates of real-world performance. Details: [DATA_SOURCES.md](DATA_SOURCES.md).

## Pipeline

```
text -> PHI detection (regex + title rule + CRF + optional high-recall backstop)
     -> disease NER (CRF with MeSH gazetteer features)
     -> assertion: ConText-style rules, plus logistic regression over context words + rule outputs
     -> MeSH ID (exact / normalised / char-n-gram TF-IDF fuzzy / NIL)
```
Assertion outputs: `present|absent|possible|hypothetical`, `current|historical`, `patient|family`, plus the triggers that fired.

## Results (full tables and error analysis in [reports/EVALUATION.md](reports/EVALUATION.md))

Real data (NCBI Disease test, 960 mentions):
* NER strict span F1 **0.80** (P 0.84, R 0.77); relaxed F1 0.91; MeSH dictionary baseline F1 0.45.
* MeSH normalisation accuracy **0.82** on mentions with a MeSH gold ID (0.53 with MeSH alone; the rest comes from adding training-split synonyms). OMIM-only mentions cannot be linked to MeSH and are reported separately.

Synthetic data (held-out names; "shifted" = phrasings unseen in training):
* Assertion accuracy, shifted: rules 0.93, hybrid 0.92, text-only ML 0.65. The ML model does not beat the rules on unseen phrasings. Rule numbers are optimistic because the same author wrote the rules and the templates.
* De-id recall, shifted: regex only 0.64, + CRF 0.88, + high-recall backstop **0.99** at precision 0.98. Remaining misses are mostly surnames that are also disease eponyms. In-distribution scores are ~1.0 and mean little.
* End-to-end on synthetic notes: NER F1 0.76 (domain shift from abstracts to note text).

## Run it

```bash
make all            # venv, download data, train (~10 min), evaluate, tests
make serve          # http://127.0.0.1:8000/docs
```
```bash
curl -s localhost:8000/extract -H 'content-type: application/json' \
  -d '{"text":"Seen by Dr. Alice Moore on 03/14/2021. Denies diabetes mellitus. Possible pneumonia. Mother had breast cancer."}'
```
Endpoints: `GET /health`, `POST /extract`, `POST /deidentify`, `POST /normalize`. Input capped at 50k characters. Docker: `docker build -t clinnlp . && docker run -p 8000:8000 clinnlp` (downloads data and trains at build time).

## Limitations
* Models and `assertion.pkl` are produced locally by `scripts/train.py`; pickle files must only be loaded from sources you trust.
* NER misses abbreviations and rare names; composite mentions are weak (recall 0.40).
* Rules handle sentence-local scope only (e.g. "No history of X in the family" is misread as patient).
* The high-recall de-id mode over-redacts; neither mode is a guarantee of de-identification.
* Only diseases are extracted (no drugs, labs, procedures).

## Layout
`src/clinnlp/` package · `scripts/` fetch, train, evaluate · `tests/` · `reports/` generated evaluation

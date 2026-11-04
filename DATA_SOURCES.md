# Data sources, licences and what is synthetic

| Asset | Origin | Licence / terms | Real or synthetic | Used for |
|---|---|---|---|---|
| NCBI Disease corpus (793 PubMed abstracts, 6,892 disease mentions with MeSH/OMIM IDs) | Doğan et al. 2014, NCBI/NLM — https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/ | Freely downloadable from NLM; PubMed abstract text is subject to publisher copyright, annotations are NLM's. Not redistributed in this repo — `make data` downloads it. | **Real** (biomedical literature, *not* clinical notes) | Train/eval of disease NER, MeSH normalisation |
| MeSH 2026 descriptors (disease tree C* and F03) | NLM — https://nlmpubs.nlm.nih.gov/projects/mesh/ | Public domain / NLM terms; downloaded by `make data` | **Real** vocabulary | Ontology lexicon (5,194 descriptors, ~59k terms), gazetteer features |
| Clinical-style notes | `src/clinnlp/synth.py` | Original code in this repo | **Synthetic** (template-generated; disease strings come from NCBI dev/test mentions) | Assertion and de-identification training/eval, end-to-end demo |
| PHI (names, dates, phones, MRNs, emails, locations, ages) | `src/clinnlp/synth.py` | Original code | **Synthetic**, randomly generated; any resemblance to real people is coincidental | De-identification training/eval |

Deliberately **not** used: MIMIC-III/IV, i2b2/n2c2, MTSamples and any other resource requiring credentialed access or with unclear redistribution terms.

Consequences you should keep in mind
* There is no public, freely redistributable corpus of *real* de-identified clinical notes with assertion and PHI annotations that avoids a data-use agreement. Assertion and de-identification numbers therefore come from synthetic notes and **do not predict performance on real notes**.
* NER and MeSH normalisation numbers are on real text, but it is PubMed abstract prose, not clinical documentation.

"""Evaluate every component and write reports/metrics.json + reports/EVALUATION.md (with error analysis).

Evaluation sets
  NER / normalisation : NCBI Disease test split (real PubMed abstracts, 100 docs, 960 mentions).
  assertion / de-id   : SYNTHETIC notes (fresh seeds, held-out names; "iid" = train phrasings, "shifted" = unseen phrasings).
  end-to-end          : SYNTHETIC "shifted" notes through the full pipeline.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinnlp import context
from clinnlp.assertion import HEADS, AssertionModel
from clinnlp.corpus import load_ncbi
from clinnlp.deid import Deidentifier, regex_spans
from clinnlp.ner import DiseaseNER, Gazetteer
from clinnlp.ontology import MeshNormalizer
from clinnlp.pipeline import Pipeline
from clinnlp.synth import Generator, disease_pool
from clinnlp.text import split_sentences, tokenize

M = ROOT / "models"
OUT = ROOT / "reports"
N_NOTES = 300


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def span_scores(gold, pred):
    """gold/pred: set of (start, end[, type]) for ONE doc. Returns strict and relaxed (any-overlap) counts."""
    g, p = set(gold), set(pred)
    tp = len(g & p)
    rg = sum(any(overlaps(x, y) for y in p) for x in g)
    rp = sum(any(overlaps(x, y) for y in g) for x in p)
    return tp, len(p) - tp, len(g) - tp, rg, len(p) - rp


def sent_of(text, pos):
    for s, e in split_sentences(text):
        if s <= pos < e:
            return text[s:e].replace("\n", " ")
    return text[max(0, pos - 60):pos + 60]


# ---------------------------------------------------------------- NER
def eval_ner(ner, docs, gaz):
    S = defaultdict(lambda: [0, 0, 0, 0, 0])
    kinds = defaultdict(lambda: [0, 0])
    fn_terms, fp_terms, bound = Counter(), Counter(), []
    base = [0, 0, 0]
    for d in docs:
        gold = [(m.start, m.end) for m in d.mentions]
        pred = [(s, e) for s, e, _, _ in ner.predict(d.text)]
        for name, vals in zip(("c", "r"), (span_scores(gold, pred),) * 2):
            pass
        tp, fp, fn, rg, rp_fp = span_scores(gold, pred)
        S["crf"] = [a + b for a, b in zip(S["crf"], (tp, fp, fn, rg, rp_fp))]
        gset, pset = set(gold), set(pred)
        for m in d.mentions:
            k = kinds[m.kind]; k[1] += 1; k[0] += (m.start, m.end) in pset
            if (m.start, m.end) not in pset:
                if any(overlaps((m.start, m.end), p) for p in pred):
                    p = next(p for p in pred if overlaps((m.start, m.end), p))
                    bound.append((m.text, d.text[p[0]:p[1]]))
                else:
                    fn_terms[m.text.lower()] += 1
        for p in pred:
            if p not in gset and not any(overlaps(p, g) for g in gold):
                fp_terms[d.text[p[0]:p[1]].lower()] += 1
        # dictionary baseline: longest MeSH-term match over tokens
        pb = []
        for s, e in split_sentences(d.text):
            toks = tokenize(d.text[s:e], s)
            fl = gaz.tag([t[0] for t in toks])
            i = 0
            while i < len(toks):
                if fl[i] == "B":
                    j = i
                    while j + 1 < len(toks) and fl[j + 1] == "I":
                        j += 1
                    pb.append((toks[i][1], toks[j][2])); i = j + 1
                else:
                    i += 1
        tpb = len(set(gold) & set(pb)); base[0] += tpb; base[1] += len(pb) - tpb; base[2] += len(gold) - tpb
    res = {"crf_strict": prf(*S["crf"][:3]),
           "crf_relaxed": prf(S["crf"][3], S["crf"][4], len(sum([[1] * len(d.mentions) for d in docs], [])) - S["crf"][3]),
           "mesh_dictionary_baseline_strict": prf(*base),
           "recall_by_kind": {k: round(v[0] / v[1], 4) for k, v in kinds.items()}}
    return res, fn_terms, fp_terms, bound


# ---------------------------------------------------------------- normalisation
def eval_norm(docs, train_syn):
    out = {}
    for name, extra in (("mesh_only", None), ("mesh_plus_train_synonyms", train_syn)):
        nz = MeshNormalizer(extra_synonyms=extra)
        c = Counter(); by_method = defaultdict(lambda: [0, 0]); errs = []
        for d in docs:
            for m in d.mentions:
                ids = set(m.concept.replace("+", "|").split("|"))
                if any(i.startswith("OMIM") for i in ids):
                    c["omim"] += 1
                    ui = nz.lookup(m.text)[0]
                    c["omim_linked_to_mesh"] += ui is not None
                    continue
                if not ids <= set(nz.names):
                    c["gold_id_not_in_current_mesh_disease_tree"] += 1
                    continue
                c["evaluable"] += 1
                ui, nm, sc, meth = nz.lookup(m.text)
                ok = ui in ids
                c["correct"] += ok
                by_method[meth][0] += ok; by_method[meth][1] += 1
                if not ok:
                    errs.append((m.text, "|".join(sorted(ids)), ui, nm, round(sc, 2), meth))
        out[name] = {"accuracy_on_evaluable": round(c["correct"] / c["evaluable"], 4), "evaluable": c["evaluable"],
                     "gold_id_outside_vocabulary": c["gold_id_not_in_current_mesh_disease_tree"], "omim_mentions": c["omim"],
                     "omim_mentions_wrongly_linked_to_mesh": c["omim_linked_to_mesh"],
                     "by_method": {k: {"n": v[1], "accuracy": round(v[0] / v[1], 4)} for k, v in by_method.items()}}
        out[name]["_errors"] = errs
    return out


# ---------------------------------------------------------------- assertion
def eval_assertion(models, notes):
    res, errs = {}, {}
    for name, fn in models.items():
        acc = {h: [0, 0] for h in HEADS}
        conf = {h: Counter() for h in HEADS}
        per_class = {h: defaultdict(lambda: [0, 0, 0]) for h in HEADS}
        e_list = []
        for n in notes:
            ents = [(e["start"], e["end"]) for e in n.entities]
            preds = fn(n.text, ents)
            for e, p in zip(n.entities, preds):
                for h in HEADS:
                    if h == "temporality" and not (e["experiencer"] == "patient" and e["assertion"] in ("present", "possible")):
                        continue
                    g, q = e[h], getattr(p, h)
                    acc[h][1] += 1; acc[h][0] += g == q
                    conf[h][(g, q)] += 1
                    pc = per_class[h]; pc[g][2] += 1; pc[q][1] += 1; pc[g][0] += g == q
                if any(getattr(p, h) != e[h] for h in ("assertion", "experiencer")):
                    e_list.append(dict(sentence=sent_of(n.text, e["start"]), entity=e["text"], gold=(e["assertion"], e["experiencer"]),
                                       pred=(p.assertion, p.experiencer), triggers=list(p.triggers)))
        r = {}
        for h in HEADS:
            f1s = {}
            for c, (tp, npred, ngold) in per_class[h].items():
                pr = tp / npred if npred else 0; rc = tp / ngold if ngold else 0
                f1s[c] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, "support": ngold}
            r[h] = {"accuracy": round(acc[h][0] / acc[h][1], 4), "n": acc[h][1], "macro_f1": round(sum(v["f1"] for v in f1s.values()) / len(f1s), 4), "per_class": f1s,
                    "confusions": {f"{g}->{q}": c for (g, q), c in conf[h].most_common() if g != q}}
        res[name] = r; errs[name] = e_list
    return res, errs


# ---------------------------------------------------------------- de-id
def eval_deid(deid, notes, mode):
    kw = dict(use_crf=mode != "regex_only", aggressive=mode == "crf_plus_aggressive")
    per = defaultdict(lambda: [0, 0, 0])
    chars = [0, 0]; fp = Counter(); misses = []; pt = [0, 0, 0]
    for n in notes:
        pred = deid.detect(n.text, **kw)
        for s, e, t in n.phi:
            per[t][2] += 1
            per[t][0] += any(overlaps((s, e), p) for p in pred)           # relaxed recall
            per[t][1] += (s, e) in {(p[0], p[1]) for p in pred}            # strict boundary
            cov = sum(max(0, min(e, p[1]) - max(s, p[0])) for p in pred)
            chars[0] += min(cov, e - s); chars[1] += e - s
            if not any(overlaps((s, e), p) for p in pred):
                misses.append((t, n.text[s:e], sent_of(n.text, s)[:140]))
        gold_sp = [(s, e) for s, e, _ in n.phi]
        for p in pred:
            if p[0] < len(n.text) and not any(overlaps((p[0], p[1]), g) for g in gold_sp):
                fp[n.text[p[0]:p[1]]] += 1
        tp_ = sum(any(overlaps((s, e), p) for p in pred) for s, e, _ in n.phi)
        pt[0] += tp_; pt[2] += len(n.phi) - tp_
        pt[1] += sum(not any(overlaps((p[0], p[1]), g) for g in gold_sp) for p in pred)
    total_gold = sum(v[2] for v in per.values())
    ents = sum(len(n.entities) for n in notes)
    r = {"overall_recall_relaxed": round(sum(v[0] for v in per.values()) / total_gold, 4),
         "overall_precision_relaxed": round(pt[0] / (pt[0] + pt[1]), 4) if pt[0] + pt[1] else 0,
         "char_level_recall": round(chars[0] / chars[1], 4),
         "by_type_recall_relaxed": {t: round(v[0] / v[2], 4) for t, v in sorted(per.items())},
         "by_type_recall_strict_boundary": {t: round(v[1] / v[2], 4) for t, v in sorted(per.items())},
         "support": {t: v[2] for t, v in sorted(per.items())}, "false_positive_spans": sum(fp.values())}
    return r, misses, fp


# ---------------------------------------------------------------- end to end
def eval_e2e(pipe, notes):
    tp = fp = fn = 0; a_ok = Counter(); a_n = Counter(); m_ok = m_n = 0; phi_hit = phi_n = phi_fp = 0
    for n in notes:
        out = pipe.extract(n.text)
        pp = [(p["start"], p["end"]) for p in out["phi"]]
        phi_n += len(n.phi); phi_hit += sum(any(overlaps((s, e), q) for q in pp) for s, e, _ in n.phi)
        phi_fp += sum(not any(overlaps(q, (s, e)) for s, e, _ in n.phi) for q in pp)
        gold = {(e["start"], e["end"]): e for e in n.entities}
        pred = {(e["start"], e["end"]): e for e in out["entities"]}
        hit = set(gold) & set(pred)
        tp += len(hit); fp += len(pred) - len(hit); fn += len(gold) - len(hit)
        for k in hit:
            g, p = gold[k], pred[k]
            for h in HEADS:
                a_n[h] += 1; a_ok[h] += g[h] == p[h]
            a_n["joint"] += 1; a_ok["joint"] += all(g[h] == p[h] for h in HEADS)
            m_n += 1; m_ok += g["mesh"] == p["mesh_id"]
    return {"ner_strict_on_notes": prf(tp, fp, fn),
            "assertion_accuracy_on_matched_spans": {h: round(a_ok[h] / a_n[h], 4) for h in a_n}, "mesh_accuracy_on_matched_spans": round(m_ok / m_n, 4),
            "n_gold_entities": tp + fn,
            "phi_in_pipeline_aggressive_with_disease_protection": {"recall_relaxed": round(phi_hit / phi_n, 4), "false_positive_spans": phi_fp, "gold_spans": phi_n}}


def md_table(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def main():
    OUT.mkdir(exist_ok=True)
    metrics = {}
    ner = DiseaseNER().load(M / "ner.crf")
    train_syn = [tuple(x) for x in json.loads((M / "train_synonyms.json").read_text())]
    test = load_ncbi("test")
    ner_res, fn_terms, fp_terms, bound = eval_ner(ner, test, Gazetteer())
    metrics["ner_ncbi_test"] = ner_res
    norm_res = eval_norm(test, train_syn)
    norm_errs = norm_res["mesh_plus_train_synonyms"].pop("_errors"); norm_res["mesh_only"].pop("_errors")
    metrics["normalization_ncbi_test_gold_mentions"] = norm_res

    pool = disease_pool()
    sets = {"iid": Generator("test", "train", seed=99, pool=pool).notes(N_NOTES), "shifted": Generator("test", "test", seed=100, pool=pool).notes(N_NOTES)}
    am = AssertionModel.load(M / "assertion.pkl")
    train_notes = Generator("train", "train", seed=1, pool=pool).notes(1500)
    am_text = AssertionModel(use_rules=False).fit(train_notes)
    models = {"rules_only (ConText)": lambda t, e: context.classify(t, e), "ml_text_only": lambda t, e: am_text.predict(t, e)[0] if e else [],
              "hybrid (ML + rule features)": lambda t, e: am.predict(t, e)[0] if e else []}
    metrics["assertion"], a_errs = {}, {}
    for s, notes in sets.items():
        metrics["assertion"][s], a_errs[s] = eval_assertion(models, notes)

    deid = Deidentifier(M / "deid.crf")
    metrics["deid"], d_misses, d_fp = {}, {}, {}
    for s, notes in sets.items():
        metrics["deid"][s] = {}
        for mode in ("regex_only", "regex_plus_crf", "crf_plus_aggressive"):
            r, misses, fp = eval_deid(deid, notes, mode)
            metrics["deid"][s][mode] = r
            d_misses[(s, mode)] = misses; d_fp[(s, mode)] = fp

    pipe = Pipeline(M)
    metrics["end_to_end_synthetic_shifted"] = eval_e2e(pipe, sets["shifted"])
    metrics["meta"] = {"n_synthetic_test_notes_per_set": N_NOTES, "synthetic_entities": {s: sum(len(n.entities) for n in ns) for s, ns in sets.items()},
                       "synthetic_phi_spans": {s: sum(len(n.phi) for n in ns) for s, ns in sets.items()}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------------- markdown report
    L = ["# Evaluation report (auto-generated by `scripts/evaluate.py`)", "",
         "Real data: NCBI Disease corpus test split (PubMed abstracts). **Synthetic**: every note used for assertion, de-identification and end-to-end numbers.",
         "Synthetic results are upper-bound-ish indicators of component behaviour, not estimates of performance on real clinical notes.", ""]
    n = metrics["ner_ncbi_test"]
    L += ["## 1. Disease NER — NCBI Disease test (real)", "",
          md_table([["MeSH dictionary longest-match (baseline)", *[n["mesh_dictionary_baseline_strict"][k] for k in ("precision", "recall", "f1")]],
                    ["CRF, strict span match", *[n["crf_strict"][k] for k in ("precision", "recall", "f1")]],
                    ["CRF, relaxed (any overlap)", *[n["crf_relaxed"][k] for k in ("precision", "recall", "f1")]]], ["system", "P", "R", "F1"]), "",
          "Strict recall by mention type: " + ", ".join(f"{k} {v}" for k, v in n["recall_by_kind"].items()), "",
          "### Error analysis", "",
          f"* Boundary errors (overlap but not exact): **{len(bound)}** of {n['crf_strict']['fn']} missed gold mentions. Examples (gold -> predicted):"]
    L += [f"  * `{g}` -> `{p}`" for g, p in bound[:10]]
    L += ["* Most frequent completely missed mentions: " + ", ".join(f"`{t}` x{c}" for t, c in fn_terms.most_common(12)),
          "* Most frequent spurious predictions (no overlap with gold): " + ", ".join(f"`{t}` x{c}" for t, c in fp_terms.most_common(12)),
          "* Reading: misses concentrate in `Modifier` mentions (e.g. adjectival uses such as 'ataxia telangiectasia gene') and rare/multiword disease names; "
          "NCBI annotation conventions on modifiers are a known source of disagreement, so precision against the gold standard is a lower bound.", ""]
    L += ["## 2. MeSH normalisation — gold mentions of NCBI test (real)", ""]
    rows = []
    for k, v in metrics["normalization_ncbi_test_gold_mentions"].items():
        rows.append([k, v["accuracy_on_evaluable"], v["evaluable"], v["gold_id_outside_vocabulary"], f'{v["omim_mentions_wrongly_linked_to_mesh"]}/{v["omim_mentions"]}',
                     "; ".join(f'{m}: {x["accuracy"]} (n={x["n"]})' for m, x in v["by_method"].items())])
    L += [md_table(rows, ["lexicon", "accuracy", "evaluable mentions", "gold ID not in disease tree", "OMIM mentions linked to MeSH (should be NIL)", "accuracy by method"]), "",
          "Composite gold IDs (`A|B`) count as correct if the prediction is any member. OMIM-only gold mentions cannot be represented in MeSH and are excluded from accuracy.",
          "The 'train synonyms' lexicon adds the *training split's* surface forms as extra synonyms (no test leakage).", "", "### Error analysis (first 15 errors, MeSH + train synonyms)", "",
          md_table([[a, b, c, d, e, f] for a, b, c, d, e, f in norm_errs[:15]], ["mention", "gold", "predicted", "predicted name", "score", "method"]), ""]
    L += ["## 3. Assertion (negation / uncertainty / hypothetical / temporality / experiencer) — SYNTHETIC notes, gold entity spans", "",
          f"{N_NOTES} notes per set. `iid` = test notes built from the same phrasing families as training (new names, new random draws). `shifted` = phrasing families never seen in training "
          "(including pseudo-negation, scope-terminator and list cases). Temporality is scored on patient entities that are present/possible.", ""]
    for s in sets:
        rows = []
        for mname, r in metrics["assertion"][s].items():
            rows.append([mname] + [f'{r[h]["accuracy"]} / {r[h]["macro_f1"]}' for h in HEADS])
        L += [f"### {s}", "", md_table(rows, ["system", "assertion acc / macro-F1", "temporality acc / macro-F1", "experiencer acc / macro-F1"]), ""]
    r = metrics["assertion"]["shifted"]["hybrid (ML + rule features)"]
    L += ["Per-class (hybrid, shifted) assertion: " + ", ".join(f"{c} F1 {v['f1']} (n={v['support']})" for c, v in r["assertion"]["per_class"].items()), "",
          "Top confusions (rules only, shifted): " + str(metrics["assertion"]["shifted"]["rules_only (ConText)"]["assertion"]["confusions"]), "",
          "### Error analysis — rules-only, shifted set (sample)", ""]
    seen = set(); k = 0
    for e in a_errs["shifted"]["rules_only (ConText)"]:
        key = (e["gold"], e["pred"], tuple(e["triggers"]))
        if key in seen: continue
        seen.add(key); k += 1
        L.append(f"* gold {e['gold']} / pred {e['pred']} — triggers {e['triggers']} — “{e['sentence'][:150]}” (entity: `{e['entity']}`)")
        if k >= 12: break
    L += ["", "### Error analysis — hybrid, shifted set (sample)", ""]
    seen = set(); k = 0
    for e in a_errs["shifted"]["hybrid (ML + rule features)"]:
        key = (e["gold"], e["pred"])
        if key in seen and k > 4: continue
        seen.add(key); k += 1
        L.append(f"* gold {e['gold']} / pred {e['pred']} — “{e['sentence'][:150]}” (entity: `{e['entity']}`)")
        if k >= 10: break
    L += ["", "## 4. PHI de-identification — SYNTHETIC notes with held-out names", "",
          "Recall is the safety-relevant number: a missed identifier is a leak. 'relaxed' = any overlap with the gold span; 'strict' = exact boundaries; char-level = share of PHI characters covered.", ""]
    for s in sets:
        rows = []
        for mode, r in metrics["deid"][s].items():
            rows.append([mode, r["overall_recall_relaxed"], r["char_level_recall"], r["overall_precision_relaxed"], r["false_positive_spans"],
                         ", ".join(f"{t} {v}" for t, v in r["by_type_recall_relaxed"].items())])
        L += [f"### {s}", "", md_table(rows, ["mode", "recall (relaxed)", "char recall", "precision (relaxed)", "false-positive spans", "recall by type"]), ""]
    L += ["Support: " + str(metrics["meta"]["synthetic_phi_spans"]), "", "### Error analysis", ""]
    for mode in ("regex_plus_crf", "crf_plus_aggressive"):
        ms = d_misses[("shifted", mode)]
        c = Counter(t for t, _, _ in ms)
        L.append(f"* **{mode}, shifted**: {len(ms)} missed spans by type {dict(c)}. Examples:")
        L += [f"  * [{t}] `{x}` in “{s}”" for t, x, s in ms[:6]]
    top = d_fp[("shifted", "crf_plus_aggressive")].most_common(12)
    L += ["* Over-redaction (false-positive spans, aggressive mode, shifted) — most frequent: " + ", ".join(f"`{t}` x{c}" for t, c in top),
          "* Name misses in aggressive mode (by token): " + ", ".join(f"`{t}` x{c}" for t, c in Counter(x for tt, x, _ in d_misses[("shifted", "crf_plus_aggressive")] if tt == "NAME").most_common(12)) +
          ". Surnames that are also in MeSH disease vocabulary (e.g. Stevens-Johnson, Edwards, Duchenne, Barrett) are treated as medical words by the backstop - the classic eponym/surname collision.", ""]
    e = metrics["end_to_end_synthetic_shifted"]
    L += ["## 5. End-to-end — SYNTHETIC shifted notes through the full pipeline", "",
          f"* NER (strict span) on notes: {e['ner_strict_on_notes']} — note the domain shift from PubMed abstracts to note-style text.",
          f"* On correctly located entities: assertion accuracy {e['assertion_accuracy_on_matched_spans']}; MeSH accuracy {e['mesh_accuracy_on_matched_spans']}.",
          f"* Gold entities: {e['n_gold_entities']}.", f"* PHI as run inside the pipeline (aggressive, disease mentions protected): {e['phi_in_pipeline_aggressive_with_disease_protection']}.", "",
          "Rule authorship caveat: the ConText trigger list was written by the project author while looking at the synthetic template vocabulary, so rule-based numbers on synthetic notes are optimistic. The `shifted` set uses different phrasings but was written by the same author, so it is a fairer signal, not an independent one.", ""]
    (OUT / "EVALUATION.md").write_text("\n".join(L))
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()

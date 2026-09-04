"""
Ingest human boundary annotations and answer the question the whole exercise exists for.

Reports, in this order:

  1. HUMAN NOISE FLOOR -- agreement between the repeated presentations of the same
     utterance. This must come first, because it is the yardstick for everything else.
     If humans disagree with each other by 30 ms, then a model at 21 ms is at the noise
     floor and the paper may say so; without this number, "21 ms" has nothing to be
     compared against.

  2. wav2vec2 VS HUMAN -- calibrates the reference the entire main results table is
     expressed against. This is the measurement that resolves (or does not) the
     reference-circularity problem.

  3. OURS VS HUMAN -- the accuracy claim we currently cannot make.

Scoring uses the same LCS word matching and pooled start/end basis as the main table, so
the numbers drop straight in.

    python scripts/23_ingest_annotations.py --dir annotation
"""

import argparse, json, unicodedata
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent


def lcs_pairs(a, b):
    A, B = len(a), len(b)
    dp = [[0]*(B+1) for _ in range(A+1)]
    for i in range(1, A+1):
        for j in range(1, B+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]["word"] == b[j-1]["word"] \
                       else max(dp[i-1][j], dp[i][j-1])
    out, i, j = [], A, B
    while i > 0 and j > 0:
        if a[i-1]["word"] == b[j-1]["word"]:
            out.append((a[i-1], b[j-1])); i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]: i -= 1
        else: j -= 1
    return out


def agree(x, y):
    e = []
    for h, r in lcs_pairs(x, y):
        e += [abs(h["start"]-r["start"])*1000, abs(h["end"]-r["end"])*1000]
    if not e: return None
    e = np.array(e)
    return {"n": len(e), "mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)),
            "w20": 100*float((e <= 20).mean()),
            "w50": 100*float((e <= 50).mean()),
            "w100": 100*float((e <= 100).mean())}


def pooled(rows):
    if not rows: return None
    tot = sum(r["n"] for r in rows)
    return {k: sum(r[k]*r["n"] for r in rows)/tot
            for k in ("mean", "median", "p90", "w20", "w50", "w100")} | {"n": tot}


def line(tag, p):
    if not p:
        print(f"  {tag:<26}{'no data':>10}"); return
    print(f"  {tag:<26}{p['n']:>8}{p['mean']:>9.1f}{p['median']:>9.1f}{p['p90']:>8.1f}"
          f"{p['w20']:>8.1f}%{p['w50']:>8.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="annotation")
    ap.add_argument("--ctc-eval",
                    default="eval_testdataset/ctc_eval_modelE_phase1/samples.jsonl")
    ap.add_argument("--ctc-key", default="modelE_oracle")
    ap.add_argument("--dtw-eval", default="eval_testdataset/ctc_eval_phase1/samples.jsonl",
                    help="file carrying the DTW baseline timestamps (key 'dtw')")
    ap.add_argument("--mfa-dir", default="mfa/aligned_word_s0.5",
                    help="IndicMFA alignment output; adds MFA-vs-human, which tests "
                         "objectively whether that alignment was degenerate")
    args = ap.parse_args()

    d = BASE/args.dir
    key = json.load(open(d/"annotation_key.json", encoding="utf-8"))

    human = {}
    files = sorted(d.glob("annotations_*.json")) + sorted(d.glob("*_annotations.json"))
    for f in files:
        blob = json.load(open(f, encoding="utf-8"))
        for u in blob.get("utterances", []):
            human[u["uid"]] = u["words"]
    if not human:
        print(f"no annotation files found in {d}. Expected annotations_*.json "
              "(downloaded, or pasted from the tool's textarea).")
        return
    print(f"loaded {len(human)} annotated presentations from {len(files)} file(s)")

    # Guard against the mojibake failure: an HTML page served without a charset
    # declaration is read as Latin-1, and every Devanagari word arrives corrupted. The
    # boundaries would still look plausible while LCS matching silently found nothing,
    # so verify the annotated words against the key before scoring anything.
    bad = []
    for uid, words in human.items():
        meta = key.get(uid)
        if not meta: continue
        # Compare NFC-normalised: the same Devanagari word can be encoded precomposed
        # (ज़ = U+095B) or decomposed (ज + nukta), and a raw string comparison flags
        # those as corrupted when they are identical text.
        nfc = lambda xs: [unicodedata.normalize("NFC", x) for x in xs]
        got = nfc([w["word"] for w in words])
        exp = nfc(meta["text"].split())
        if got and exp and got != exp:
            bad.append((uid, exp[:3], got[:3]))
    if bad:
        print(f"\n  *** {len(bad)} of {len(human)} annotations have words that do not match "
              "the reference ***")
        for uid, exp, got in bad[:3]:
            print(f"      {uid}: expected {exp} but got {got}")
        print("      Most likely cause: the HTML was opened without a UTF-8 charset, so the "
              "text was mangled.\n      These annotations cannot be scored -- regenerate the "
              "batches and redo them.\n")
        if len(bad) == len(human):
            print("  all annotations affected; nothing to score."); return
    print()

    print(f"  {'comparison':<26}{'bounds':>8}{'mean':>9}{'median':>9}{'p90':>8}"
          f"{'<=20ms':>9}{'<=50ms':>9}")
    print("  " + "-"*70)

    # 1. human noise floor, from repeated presentations
    # copies of one utterance now sit in different batches (different annotators), so
    # pair them by corpus idx: this is INTER-annotator agreement, the real noise floor.
    by_idx = {}
    for uid, meta in key.items():
        if uid in human:
            by_idx.setdefault(meta["idx"], []).append(human[uid])
    floor = []
    for idx, anns in by_idx.items():
        if len(anns) >= 2:
            r = agree(anns[0], anns[1])
            if r: floor.append(r)
    fp = pooled(floor)
    line("human vs human (INTER-ann.)", fp)

    # 2 & 3. models against the human reference
    ctc = {}
    p = BASE/args.ctc_eval
    if p.exists():
        for l in open(p, encoding="utf-8"):
            s = json.loads(l); ctc[s["idx"]] = s

    # MFA, keyed by the same corpus idx. We previously judged this alignment degenerate
    # by inference (our head and wav2vec2 agreed at 21 ms while MFA differed from both by
    # ~415 ms). Human boundaries turn that inference into a measurement.
    # The shipped IndicMFA dictionary is grapheme-to-grapheme, so the original run fed
    # MFA a grapheme-tokenised transcript and it treated every grapheme as a word --
    # free to insert silence INSIDE words. mfa/aligned_word_s0.5 is the corrected run
    # (word-level lexicon, MFA's own silence model), which is the fair number to report.
    mfa_by_idx = {}
    mdir = BASE/args.mfa_dir
    if mdir.exists():
        import importlib.util as _ilu, sys as _sys
        _sp = _ilu.spec_from_file_location("sc", BASE/"scripts"/"24_score_mfa_variant.py")
        _sc = _ilu.module_from_spec(_sp)
        _av, _sys.argv = _sys.argv, ["x"]; _sp.loader.exec_module(_sc); _sys.argv = _av
        mmeta = json.load(open(BASE/"mfa/corpus_meta.json", encoding="utf-8"))
        for muid, mm in mmeta.items():
            tg = mdir/f"{muid}.TextGrid"
            if not tg.exists(): continue
            mw = _sc.word_tier(tg, mm)
            if mw: mfa_by_idx[mm["idx"]] = mw

    # DTW baseline, keyed by the same corpus idx. Its 107.2 ms in the main table is
    # agreement with wav2vec2; against human boundaries it can be read as accuracy.
    dtw = {}
    pd = BASE/args.dtw_eval
    if pd.exists():
        for l in open(pd, encoding="utf-8"):
            t = json.loads(l)
            w = (t.get("dtw") or {}).get("words") or []
            if w: dtw[t["idx"]] = w

    w2v_rows, our_rows, mfa_rows, dtw_rows = [], [], [], []
    seen_idx = set()
    for uid, words in human.items():
        meta = key.get(uid)
        if not meta: continue
        if meta["idx"] in seen_idx: continue     # score each utterance once
        seen_idx.add(meta["idx"])
        s = ctc.get(meta["idx"])
        if not s: continue
        vw = (s.get("wav2vec2") or {}).get("words") or []
        ow = (s.get(args.ctc_key) or {}).get("words") or []
        if vw:
            r = agree(vw, words)
            if r: w2v_rows.append(r)
        if ow:
            r = agree(ow, words)
            if r: our_rows.append(r)
        mw = mfa_by_idx.get(meta["idx"])
        if mw:
            r = agree(mw, words)
            if r: mfa_rows.append(r)
        dw = dtw.get(meta["idx"])
        if dw:
            r = agree(dw, words)
            if r: dtw_rows.append(r)

    wp, op, mp = pooled(w2v_rows), pooled(our_rows), pooled(mfa_rows)
    dp = pooled(dtw_rows)
    line("wav2vec2 vs human", wp)
    line("OURS vs human", op)
    line("DTW baseline vs human", dp)
    line("IndicMFA (fixed) vs human", mp)

    print()
    if fp and wp:
        print(f"  wav2vec2 is {wp['mean']/max(fp['mean'],1e-9):.1f}x the human noise floor")
    if fp and op:
        print(f"  our head is {op['mean']/max(fp['mean'],1e-9):.1f}x the human noise floor")
    if op and wp:
        verdict = ("our head is CLOSER to human than wav2vec2 is"
                   if op["mean"] < wp["mean"] else
                   "wav2vec2 is closer to human than our head is")
        print(f"  {verdict} ({op['mean']:.1f} vs {wp['mean']:.1f} ms)")
    if fp and op and op["mean"] <= fp["mean"]*1.2:
        print("  our head is AT the human noise floor: no headroom remains to measure.")

    if mp and op:
        print(f"  IndicMFA is {mp['mean']/max(op['mean'],1e-9):.1f}x our head's distance "
              "from human" + ("  -> the earlier 'MFA is degenerate' reading is CONFIRMED"
                              if mp["mean"] > 2*op["mean"] else
                              "  -> MFA is NOT clearly degenerate; revisit that conclusion"))

    if dp and op:
        print(f"  DTW is {dp['mean']/max(op['mean'],1e-9):.1f}x our head's distance from human")
    json.dump({"human_floor": fp, "wav2vec2_vs_human": wp, "ours_vs_human": op,
               "dtw_vs_human": dp, "mfa_vs_human": mp,
               "n_annotated": len(human), "n_repeat_pairs": len(floor)},
              open(d/"human_comparison.json", "w"), indent=2)
    print(f"\nsaved -> {d/'human_comparison.json'}")


if __name__ == "__main__":
    main()

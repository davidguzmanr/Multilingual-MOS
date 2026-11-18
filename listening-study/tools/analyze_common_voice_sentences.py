"""Summarise where each locale's Common Voice sentences come from (wiki vs sentence-collector vs other) and the
sentence_domain tags, using the TSVs inside an extracted release. Prints a table and random non-wiki examples.
    python listening-study/tools/analyze_common_voice_sentences.py [locale ...]
"""
import glob, os, sys, collections, random
import pandas as pd
ROOT = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS/data/Common-Voice"
def find(loc, name):
    hits = glob.glob(f"{ROOT}/{loc}/**/{name}", recursive=True); return hits[0] if hits else None
locales = sys.argv[1:] or sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}") and glob.glob(f"{ROOT}/{d}/**/*.tsv", recursive=True))
for loc in locales:
    vs = find(loc, "validated_sentences.tsv"); val = find(loc, "validated.tsv")
    if not vs and not val: print(f"== {loc}: no TSVs found (not extracted?)"); continue
    print(f"\n== {loc}")
    if vs:
        s = pd.read_csv(vs, sep="\t", quoting=3, dtype=str, on_bad_lines="skip")
        print(f"   validated_sentences.tsv: {len(s):,} sentences; columns: {list(s.columns)}")
        if "source" in s: print("   source:", dict(collections.Counter(s["source"].fillna("?")).most_common(8)))
        if "sentence_domain" in s: print("   sentence_domain:", dict(collections.Counter(s["sentence_domain"].fillna("(none)")).most_common(10)))
        nonwiki = s[~s.get("source", pd.Series([""]*len(s))).fillna("").str.contains("wiki", case=False)] if "source" in s else s
        for t in random.Random(0).sample(list(nonwiki["sentence"].dropna()), min(6, len(nonwiki))): print("     ·", t[:120])
    if val:
        v = pd.read_csv(val, sep="\t", quoting=3, dtype=str, on_bad_lines="skip", usecols=lambda c: c in ("client_id","sentence","sentence_domain","age","gender"))
        print(f"   validated.tsv: {len(v):,} clips, {v['client_id'].nunique():,} speakers; top speaker has {v['client_id'].value_counts().iloc[0]:,} clips")

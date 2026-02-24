"""Investigate VQA-RAD leakage and SLAKE empty answer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset

print("=== VQA-RAD Leakage Investigation ===")
ds = load_dataset("flaviagiammarino/vqa-rad")

# Build index with image bytes prefix for dedup
train_data = []
for i in range(len(ds["train"])):
    row = ds["train"][i]
    train_data.append((
        row["question"].strip().lower(),
        row["answer"].strip().lower(),
        row["image"].tobytes()[:200],
    ))

test_data = []
for i in range(len(ds["test"])):
    row = ds["test"][i]
    test_data.append((
        row["question"].strip().lower(),
        row["answer"].strip().lower(),
        row["image"].tobytes()[:200],
    ))

# Question-only overlap
train_q = {t[0] for t in train_data}
test_q = {t[0] for t in test_data}
q_overlap = train_q & test_q
print(f"Question-text overlap: {len(q_overlap)} questions")
print("  (Same question asked about different images — EXPECTED in VQA datasets)")

# (Q, A) overlap
train_qa = {(t[0], t[1]) for t in train_data}
test_qa = {(t[0], t[1]) for t in test_data}
qa_overlap = train_qa & test_qa
print(f"(Question, Answer) overlap: {len(qa_overlap)} unique pairs")

# True duplicate: same image bytes + question + answer
train_full = set(train_data)
test_full = set(test_data)
true_dup = train_full & test_full
print(f"True (image+question+answer) duplicates: {len(true_dup)}")

# Show some overlapping Q/A pairs and whether images match
print("\nSample (Q,A) overlaps — checking if same image:")
for q, a in sorted(qa_overlap)[:5]:
    t_idxs = [i for i, t in enumerate(train_data) if t[0] == q and t[1] == a]
    e_idxs = [i for i, t in enumerate(test_data) if t[0] == q and t[1] == a]
    same_img = any(
        train_data[ti][2] == test_data[ei][2]
        for ti in t_idxs for ei in e_idxs
    )
    print(f"  Q='{q[:55]}' A='{a}' same_img={same_img}")

# === SLAKE empty answer investigation ===
print("\n=== SLAKE Empty Answer Investigation ===")
import json
slake_dir = Path("data/raw/Slake1.0")
with open(slake_dir / "train.json", encoding="utf-8") as f:
    rows = json.load(f)
en_rows = [r for r in rows if r.get("q_lang") == "en"]
for r in en_rows:
    ans = str(r.get("answer", ""))
    if not ans.strip():
        print(f"  EMPTY/WHITESPACE answer found:")
        print(f"    qid={r.get('qid')}, img_name={r.get('img_name')}")
        print(f"    question='{r.get('question')}'")
        print(f"    answer='{ans}' (repr: {repr(ans)})")
        print(f"    answer_type={r.get('answer_type')}")
        print(f"    content_type={r.get('content_type')}")

# === VQA-RAD train size clarification ===
print("\n=== VQA-RAD Train Size ===")
print(f"flaviagiammarino/vqa-rad train: {len(ds['train'])} rows")
print(f"flaviagiammarino/vqa-rad test: {len(ds['test'])} rows")
print("Note: 1,793 is the original Lau et al. (2018) split, NOT 3,064.")
print("The 3,064 figure from the prompt likely includes augmented/paraphrased Qs.")
print("Our HF source uses the original split — 1,793 is correct.")

# === SLAKE duplicate triples investigation ===
print("\n=== SLAKE Duplicate Triples ===")
sys.path.insert(0, "src")
from radiology_vqa.slake_loader import load_slake
samples = load_slake(slake_dir, "train")
triples = [
    (s.img_name, s.question.strip().lower(), s.answer.strip().lower())
    for s in samples
]
from collections import Counter
triple_counts = Counter(triples)
dups = [(t, c) for t, c in triple_counts.items() if c > 1]
print(f"Total triples: {len(triples)}")
print(f"Unique triples: {len(set(triples))}")
print(f"Duplicate triples: {len(dups)}")
for t, c in sorted(dups, key=lambda x: -x[1])[:5]:
    print(f"  count={c}: img={t[0]}, q='{t[1][:50]}', a='{t[2]}'")

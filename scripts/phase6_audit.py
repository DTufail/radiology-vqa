"""Phase 6A Dataset Audit — memory-optimized version."""
import gc
import json
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiology_vqa.config import Settings

settings = Settings()


def print_sep(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def sample_metrics(data, label):
    """Compute all metrics in one pass to avoid multiple iterations."""
    if not data:
        return
    s0 = data[0]
    print(f"  Count: {len(data)}")
    print(f"  Type: {type(s0).__name__}")
    print(f"  Fields: {list(type(s0).model_fields.keys())}")  # class-level access (no deprecation warning)
    print(f"  Image type: {type(s0.image).__name__}, mode: {getattr(s0.image, 'mode', 'N/A')}")
    print(f"  Question: {s0.question[:80]}")
    print(f"  Answer: {s0.answer}")
    print(f"  answer_type: {s0.answer_type}, modality: {s0.modality}")
    print(f"  source: {s0.source}, sample_id: {s0.sample_id}")

    # Single pass over data
    type_ctr, mod_ctr = Counter(), Counter()
    empty_q = empty_a = non_rgb = html_count = long_50 = ws_only = non_str = 0
    ans_lens = []
    unique_answers = set()
    qa_pairs = set()
    img_sample_sizes = []
    sample_indices = set(random.sample(range(len(data)), min(100, len(data))))

    for i, s in enumerate(data):
        type_ctr[s.answer_type] += 1
        mod_ctr[s.modality] += 1
        if not s.question or not s.question.strip():
            empty_q += 1
        ans = s.answer
        if not isinstance(ans, str):
            non_str += 1
            ans = str(ans)
        if not ans.strip():
            ws_only += 1
        elif not ans:
            empty_a += 1
        words = len(ans.split())
        ans_lens.append(words)
        if words > 50:
            long_50 += 1
        if "<" in ans and ">" in ans:
            html_count += 1
        if getattr(s.image, "mode", None) != "RGB":
            non_rgb += 1
        unique_answers.add(ans.strip().lower())
        qa_pairs.add((s.question.strip().lower(), ans.strip().lower()))
        if i in sample_indices and hasattr(s.image, "size"):
            img_sample_sizes.append(s.image.size)

    sorted_lens = sorted(ans_lens)
    n = len(ans_lens)
    print(f"  answer_type dist: {dict(type_ctr)}")
    print(f"  modality dist: {dict(mod_ctr)}")
    print(f"  Empty questions: {empty_q} | Empty answers: {empty_a}")
    print(f"  Answer len: mean={sum(ans_lens)/n:.1f}, median={sorted_lens[n//2]}, "
          f"max={max(ans_lens)}, pct>10words={sum(1 for l in ans_lens if l > 10)/n*100:.1f}%")
    print(f"  Non-RGB images: {non_rgb}")
    print(f"  Unique answers: {len(unique_answers)}")
    print(f"  Duplicate (question,answer) pairs: {len(data) - len(qa_pairs)}")
    print(f"  Normalization — html_tags={html_count}, >50words={long_50}, "
          f"whitespace_only={ws_only}, non_string={non_str}")
    if img_sample_sizes:
        widths = [w for w, h in img_sample_sizes]
        heights = [h for w, h in img_sample_sizes]
        print(f"  Image sizes (sample 100): width [{min(widths)}-{max(widths)}], "
              f"height [{min(heights)}-{max(heights)}], unique: {len(set(img_sample_sizes))}")

    return {
        "questions": {s.question.strip().lower() for s in data},
        "qa_pairs": qa_pairs,
    }


# ── STEP 2A: VQA-RAD ────────────────────────────────────────────────────
print_sep("STEP 2A: VQA-RAD — HuggingFace raw dataset")
try:
    from datasets import load_dataset
    ds_vqarad = load_dataset("flaviagiammarino/vqa-rad")
    print(f"Available splits: {list(ds_vqarad.keys())}")
    for split_name in ds_vqarad:
        split = ds_vqarad[split_name]
        print(f"  {split_name}: {len(split)} rows, columns: {split.column_names}")
        row0 = split[0]
        print(f"    image mode: {row0['image'].mode}, question: {row0['question'][:80]}")
        print(f"    answer: {row0['answer']}")
    del ds_vqarad
    gc.collect()
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# ── STEP 2B: PathVQA ────────────────────────────────────────────────────
print_sep("STEP 2B: PathVQA — HuggingFace raw dataset")
try:
    from datasets import load_dataset
    ds_pathvqa = load_dataset("flaviagiammarino/path-vqa", streaming=True)
    print(f"Available splits: {list(ds_pathvqa.keys())}")
    non_rgb = 0
    for i, row in enumerate(ds_pathvqa["train"]):
        if i == 0:
            print(f"  train columns: {list(row.keys())}")
        if i >= 500:
            break
        if row["image"].mode != "RGB":
            non_rgb += 1
    print(f"  Non-RGB in first 500 train: {non_rgb}")
    del ds_pathvqa
    gc.collect()
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# ── STEP 2C: SLAKE ──────────────────────────────────────────────────────
print_sep("STEP 2C: SLAKE — Local JSON verification")
slake_dir = settings.slake_dir
if slake_dir.exists():
    for fname in ["train.json", "validate.json", "test.json"]:
        fpath = slake_dir / fname
        if not fpath.exists():
            print(f"  MISSING: {fpath}")
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        en = [d for d in data if d.get("q_lang") == "en"]
        print(f"  {fname}: {len(data)} total, {len(en)} English")
        if data:
            print(f"    Keys: {list(data[0].keys())}")
            mods = Counter(d.get("modality", "?") for d in en)
            atypes = Counter(str(d.get("answer_type", "?")).upper() for d in en)
            print(f"    Modalities (EN): {dict(mods)}")
            print(f"    Answer types (EN): {dict(atypes)}")
        missing_imgs = [
            d.get("img_name", "") for d in en
            if not (slake_dir / "imgs" / d.get("img_name", "")).exists()
        ]
        if missing_imgs:
            print(f"    MISSING IMAGES: {len(missing_imgs)} — {missing_imgs[:5]}")
        else:
            print(f"    All {len(en)} image references OK")
        del data, en
        gc.collect()
else:
    print(f"  SLAKE dir not found: {slake_dir}")

# ── STEP 3 & 4 & 5: Loaders — process one at a time ────────────────────
print_sep("STEP 3+4+5: Loaders, Leakage & Readiness (memory-efficient)")

from radiology_vqa.loader import load_pathvqa, load_vqa_rad
from radiology_vqa.slake_loader import load_slake

leakage_index = {}

# All entries are 4-tuples: (label, fn, positional_args, keyword_args)
loader_configs = [
    ("VQA-RAD train",     load_vqa_rad,  ("train",),                        {}),
    ("VQA-RAD test",      load_vqa_rad,  ("test",),                         {}),
    ("SLAKE train",       load_slake,    (settings.slake_dir, "train"),      {}),
    ("SLAKE validation",  load_slake,    (settings.slake_dir, "validation"), {}),
    ("SLAKE test",        load_slake,    (settings.slake_dir, "test"),       {}),
    ("PathVQA train",     load_pathvqa,  ("train",),                        {"max_samples": 500}),
]

for label, loader_fn, args, kwargs in loader_configs:
    print(f"\n--- {label} ---")
    try:
        data = loader_fn(*args, **kwargs)
        metrics = sample_metrics(data, label)
        if metrics:
            leakage_index[label] = metrics
        del data
        gc.collect()
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

# ── STEP 4: Leakage using saved index sets ───────────────────────────────
print_sep("STEP 4: Data Leakage Check")

vqa_train_idx = leakage_index.get("VQA-RAD train", {})
vqa_test_idx  = leakage_index.get("VQA-RAD test", {})
slake_idx     = leakage_index.get("SLAKE train", {})
pathvqa_idx   = leakage_index.get("PathVQA train", {})

if vqa_train_idx and vqa_test_idx:
    overlap = vqa_test_idx["questions"] & vqa_train_idx["questions"]
    print(f"VQA-RAD train/test question overlap: {len(overlap)}")
    if overlap:
        print("  *** OVERLAP DETECTED ***")
        for q in sorted(overlap)[:10]:
            print(f"    - {q[:80]}")
    else:
        print("  PASS — no overlap")
    qa_overlap = vqa_test_idx["qa_pairs"] & vqa_train_idx["qa_pairs"]
    print(f"VQA-RAD train/test (question,answer) overlap: {len(qa_overlap)}")
else:
    print("  SKIPPED — VQA-RAD train or test not loaded")

for other_label, other_idx in [("SLAKE train", slake_idx), ("PathVQA train", pathvqa_idx)]:
    if other_idx and vqa_test_idx:
        overlap = vqa_test_idx["questions"] & other_idx["questions"]
        print(f"{other_label} / VQA-RAD test question overlap: {len(overlap)}")
        if overlap:
            for q in sorted(overlap)[:10]:
                print(f"    - {q[:80]}")
    else:
        print(f"  SKIPPED — {other_label} or VQA-RAD test not loaded")

# ── STEP 5a/5b: Quick checks ─────────────────────────────────────────────
print_sep("STEP 5: Phase 6 Readiness Gaps")

print("\n5a: Unified loader (load_all)?")
try:
    from radiology_vqa.loader import load_all
    print("  load_all() EXISTS")
except ImportError:
    print("  load_all() DOES NOT EXIST — needs to be built")

print("\n5b: HuggingFace Dataset bridge?")
try:
    from datasets import Dataset
    print("  datasets library available — Dataset class importable")
except ImportError:
    print("  datasets library NOT available — install needed")

print_sep("AUDIT COMPLETE")
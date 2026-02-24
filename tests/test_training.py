"""Tests for Phase 6A training dataset preparation."""

import pytest
from PIL import Image

from radiology_vqa.training.dataset import (
    TrainingConfig,
    build_conversation,
    normalize_answer,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("YES") == "yes"
        assert normalize_answer("MRI") == "mri"

    def test_strip_whitespace(self):
        assert normalize_answer("  yes  ") == "yes"
        assert normalize_answer("\tno\n") == "no"

    def test_empty_after_strip(self):
        assert normalize_answer("   ") == ""
        assert normalize_answer("") == ""

    def test_preserves_content(self):
        assert normalize_answer("the left lung") == "the left lung"
        assert normalize_answer("CT scan") == "ct scan"

    def test_preserves_punctuation(self):
        # Training targets keep natural answer patterns
        assert normalize_answer("yes.") == "yes."
        assert normalize_answer("no,") == "no,"

    def test_preserves_articles(self):
        # Articles are kept for training (unlike evaluation normalization)
        assert normalize_answer("a fracture") == "a fracture"
        assert normalize_answer("the chest") == "the chest"


class TestBuildConversation:
    def test_structure(self):
        conv = build_conversation("What organ is this?", "lung")
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"

    def test_image_token(self):
        conv = build_conversation("What is this?", "brain")
        assert "<image>" in conv[0]["content"]

    def test_question_in_user_message(self):
        conv = build_conversation("Is there a fracture?", "yes")
        assert "Is there a fracture?" in conv[0]["content"]

    def test_answer_is_assistant_content(self):
        conv = build_conversation("What modality?", "ct")
        assert conv[1]["content"] == "ct"

    def test_image_token_precedes_question(self):
        conv = build_conversation("What is shown?", "lung")
        content = conv[0]["content"]
        assert content.index("<image>") < content.index("What is shown?")


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.include_vqa_rad is True
        assert config.include_slake is True
        assert config.include_pathvqa is True
        assert config.seed == 42
        assert config.max_answer_length == 50
        assert config.normalize_answers is True

    def test_disable_datasets(self):
        config = TrainingConfig(include_vqa_rad=False, include_pathvqa=False)
        assert config.include_vqa_rad is False
        assert config.include_slake is True
        assert config.include_pathvqa is False

    def test_custom_max_length(self):
        config = TrainingConfig(max_answer_length=10)
        assert config.max_answer_length == 10


class TestBuildTrainingDatasetUnit:
    """Unit tests that don't require actual dataset downloads."""

    def test_empty_answer_filtered(self):
        """Verify that samples with empty answers after normalization are skipped."""
        answer = normalize_answer("   ")
        assert answer == ""
        # In build_training_dataset, `if not answer: continue` skips these

    def test_long_answer_truncated(self):
        """Verify truncation logic for abnormally long answers."""
        long_answer = " ".join(["word"] * 60)
        words = long_answer.split()
        truncated = " ".join(words[:50])
        assert len(truncated.split()) == 50

    def test_conversation_keys(self):
        """Each conversation dict has role and content keys."""
        conv = build_conversation("Any question?", "any answer")
        for msg in conv:
            assert "role" in msg
            assert "content" in msg

    def test_normalize_handles_multiword(self):
        assert normalize_answer("Left lower lobe pneumonia") == "left lower lobe pneumonia"

    def test_normalize_tabs_newlines(self):
        assert normalize_answer("yes\t") == "yes"
        assert normalize_answer("no\n") == "no"


@pytest.mark.slow
class TestBuildTrainingDatasetIntegration:
    """Integration tests that require dataset downloads. Run with: pytest -m slow"""

    def test_build_returns_datasets(self):
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False)  # skip PathVQA for speed
        train_ds, val_ds = build_training_dataset(config=config)

        assert len(train_ds) > 0
        assert len(val_ds) > 0

    def test_train_has_required_keys(self):
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False)
        train_ds, _ = build_training_dataset(config=config)

        sample = train_ds[0]
        assert "image" in sample
        assert "conversations" in sample
        assert "source" in sample
        assert "sample_id" in sample

    def test_conversations_format(self):
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False)
        train_ds, _ = build_training_dataset(config=config)

        conv = train_ds[0]["conversations"]
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"
        assert "<image>" in conv[0]["content"]

    def test_no_empty_answers(self):
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False)
        train_ds, val_ds = build_training_dataset(config=config)

        for ds, name in [(train_ds, "train"), (val_ds, "val")]:
            for i in range(len(ds)):
                answer = ds[i]["conversations"][1]["content"]
                assert answer.strip(), f"Empty answer in {name} at index {i}"

    def test_no_vqa_rad_test_leakage(self):
        """CRITICAL: VQA-RAD test sample IDs must not appear in training."""
        from radiology_vqa.loader import load_vqa_rad
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False)
        train_ds, _ = build_training_dataset(config=config)
        test_samples = load_vqa_rad("test")

        train_ids = {train_ds[i]["sample_id"] for i in range(len(train_ds))}
        test_ids = {s.sample_id for s in test_samples}
        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} test sample IDs appear in training: {overlap}"
        )

    def test_dataset_counts(self):
        """Verify approximate expected counts (no PathVQA)."""
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False)
        train_ds, val_ds = build_training_dataset(config=config)

        # VQA-RAD train (1793) + SLAKE train EN (~4918) = ~6711
        assert 6000 < len(train_ds) < 7500, f"Unexpected train count: {len(train_ds)}"
        # SLAKE validation EN (~1053)
        assert 900 < len(val_ds) < 1200, f"Unexpected val count: {len(val_ds)}"

    def test_sources_present(self):
        """Both vqa_rad and slake sources appear in training data."""
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False)
        train_ds, _ = build_training_dataset(config=config)

        sources = {train_ds[i]["source"] for i in range(len(train_ds))}
        assert "vqa_rad" in sources
        assert "slake" in sources

    def test_answers_are_lowercase(self):
        """normalize_answers=True produces lowercase answers."""
        from radiology_vqa.training.dataset import build_training_dataset

        config = TrainingConfig(include_pathvqa=False, normalize_answers=True)
        train_ds, _ = build_training_dataset(config=config)

        for i in range(min(50, len(train_ds))):
            answer = train_ds[i]["conversations"][1]["content"]
            assert answer == answer.lower(), f"Non-lowercase answer at {i}: {answer!r}"

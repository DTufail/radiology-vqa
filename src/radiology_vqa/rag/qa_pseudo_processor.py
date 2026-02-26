"""QA Pseudo-Document Processor.

Converts VQA-RAD and SLAKE training QA pairs into Document objects for indexing.

Each training QA pair becomes a document with text:
    "Question: {question} Answer: {answer}"

SLAKE documents are additionally enriched with available metadata:
    " Body region: {location} Modality: {modality} Category: {content_type}"

These pseudo-documents give BM25 exact-term matching for question-level retrieval
during inference, directly targeting the over-abstention failure mode where the
supervisor rejects a correct answer due to insufficient retrieved evidence.

Language filter: SLAKE train.json contains both English and Chinese entries.
Only English entries (q_lang == 'en') are included. Chinese entries would corrupt
BM25 tokenisation (whitespace splitting produces single Unicode characters, not words).
"""

import json
import logging
from pathlib import Path

from radiology_vqa.rag.document import Document, DocumentMeta

logger = logging.getLogger(__name__)


class QAPseudoProcessor:
    """Convert VQA-RAD and SLAKE training QA pairs into retrieval documents.

    Args:
        slake_train_path: Path to SLAKE train.json (local file).
    """

    def __init__(self, slake_train_path: str) -> None:
        self._slake_train_path = slake_train_path

    def process_vqarad(self, dataset=None) -> list[Document]:
        """Load VQA-RAD train split and return pseudo-documents.

        Args:
            dataset: Pre-loaded dataset (iterable of dicts with 'question' and
                     'answer' keys). If None, loads from HuggingFace. Pass a
                     mock dataset in tests to avoid network calls.

        Returns:
            List of Document objects with source_type='qa_vqarad'.
        """
        if dataset is None:
            from datasets import load_dataset

            dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")

        documents: list[Document] = []
        for idx, row in enumerate(dataset):
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue

            text = f"Question: {question} Answer: {answer}"

            documents.append(
                Document(
                    text=text,
                    meta=DocumentMeta(
                        source_type="qa_vqarad",
                        entity_name=f"vqarad_{idx}",
                        attribute="qa_pair",
                        source_file="vqa_rad",
                        chunk_index=0,
                    ),
                    doc_id=f"qa_vqarad_{idx}",
                )
            )

        logger.info("QAPseudoProcessor: %d VQA-RAD pseudo-documents", len(documents))
        return documents

    def process_slake(self) -> list[Document]:
        """Load SLAKE train.json (English only) and return pseudo-documents.

        Filters strictly to q_lang == 'en'. Appends location, modality, and
        content_type metadata when available.

        Returns:
            List of Document objects with source_type='qa_slake'.
        """
        with open(self._slake_train_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        documents: list[Document] = []
        skipped_non_english = 0

        for entry in entries:
            # Strict English-only filter
            if entry.get("q_lang", "").lower() != "en":
                skipped_non_english += 1
                continue

            question = str(entry.get("question", "")).strip()
            answer = str(entry.get("answer", "")).strip()
            if not question or not answer:
                continue

            # Base content
            text = f"Question: {question} Answer: {answer}"

            # Metadata enrichment
            location = str(entry.get("location") or "").strip()
            modality = str(entry.get("modality") or "").strip()
            content_type = str(entry.get("content_type") or "").strip()

            if location:
                text += f" Body region: {location}"
            if modality:
                text += f" Modality: {modality}"
            if content_type:
                text += f" Category: {content_type}"

            qid = entry.get("qid", len(documents))

            documents.append(
                Document(
                    text=text,
                    meta=DocumentMeta(
                        source_type="qa_slake",
                        entity_name=f"slake_{qid}",
                        attribute="qa_pair",
                        source_file=Path(self._slake_train_path).name,
                        chunk_index=0,
                    ),
                    doc_id=f"qa_slake_{qid}",
                )
            )

        logger.info(
            "QAPseudoProcessor: %d SLAKE pseudo-documents (skipped %d non-English)",
            len(documents),
            skipped_non_english,
        )
        return documents

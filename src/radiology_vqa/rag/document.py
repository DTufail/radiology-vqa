import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentMeta(BaseModel):
    """Provenance metadata for a single indexed document chunk."""

    source_type: str  # "kg_disease", "kg_organ", "kg_organ_rel", "pubmed", "slake_qa"
    entity_name: str  # disease/organ name, or PubMed article ID
    attribute: str  # "symptom", "treatment", "function", "summary", "abstract", etc.
    source_file: str  # originating file name
    chunk_index: int = 0  # chunk position within parent document (0 if not chunked)


class Document(BaseModel):
    """A single chunk of text ready for indexing, with full provenance."""

    text: str = Field(min_length=1)
    meta: DocumentMeta
    doc_id: str  # unique: "{source_type}_{entity}_{attribute}_{chunk_index}"


class RetrievalResult(BaseModel):
    """Single retrieval result returned to the caller."""

    document: Document
    score: float
    rank: int  # 1-indexed position in results

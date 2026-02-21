import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Split long text into overlapping chunks using word-based splitting."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """chunk_size and chunk_overlap are in words (whitespace-split tokens)."""
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks.
        Texts shorter than chunk_size return as single-element list.
        Empty/whitespace-only text returns empty list."""
        if not text or not text.strip():
            return []

        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(chunk_text)
            if end == len(words):
                break
            start += step

        return chunks

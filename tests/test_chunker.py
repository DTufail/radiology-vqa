import pytest

from radiology_vqa.rag.chunker import TextChunker


def test_short_text_single_chunk():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    result = chunker.chunk("This is a short sentence.")
    assert len(result) == 1
    assert result[0] == "This is a short sentence."


def test_long_text_multiple_chunks():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    words = ["word"] * 30
    text = " ".join(words)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 10


def test_adjacent_chunks_share_overlap():
    chunker = TextChunker(chunk_size=10, chunk_overlap=3)
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Last 3 words of chunk 0 should appear at start of chunk 1
    end_words = chunks[0].split()[-3:]
    start_words = chunks[1].split()[:3]
    assert end_words == start_words


def test_empty_string_returns_empty():
    chunker = TextChunker()
    assert chunker.chunk("") == []


def test_whitespace_only_returns_empty():
    chunker = TextChunker()
    assert chunker.chunk("   \n\t  ") == []


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)

import numpy as np
import pytest


@pytest.mark.slow
def test_embedder_initializes():
    from radiology_vqa.rag.embedder import Embedder

    embedder = Embedder()
    assert embedder.dimension > 0
    assert isinstance(embedder.model_name, str)


@pytest.mark.slow
def test_embed_texts_shape():
    from radiology_vqa.rag.embedder import Embedder

    embedder = Embedder()
    result = embedder.embed_texts(["test sentence"])
    assert result.shape == (1, embedder.dimension)
    assert result.dtype == np.float32


@pytest.mark.slow
def test_embed_query_shape():
    from radiology_vqa.rag.embedder import Embedder

    embedder = Embedder()
    result = embedder.embed_query("test query")
    assert result.shape == (1, embedder.dimension)
    assert result.dtype == np.float32


@pytest.mark.slow
def test_embeddings_are_l2_normalized():
    from radiology_vqa.rag.embedder import Embedder

    embedder = Embedder()
    vecs = embedder.embed_texts(["one sentence", "another sentence", "third one"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.01)


@pytest.mark.slow
def test_semantic_similarity():
    from radiology_vqa.rag.embedder import Embedder

    embedder = Embedder()
    e1 = embedder.embed_texts(["lung disease symptoms"])
    e2 = embedder.embed_texts(["pulmonary condition signs"])
    e3 = embedder.embed_texts(["knee replacement surgery"])

    sim_medical = float(np.dot(e1[0], e2[0]))
    sim_unrelated = float(np.dot(e1[0], e3[0]))
    assert sim_medical > sim_unrelated, (
        f"Expected sim(lung disease, pulmonary condition) > sim(lung disease, knee surgery):"
        f" {sim_medical:.4f} vs {sim_unrelated:.4f}"
    )

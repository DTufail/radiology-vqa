import json

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# MockEmbedder — zero-dependency mock for fast (non-slow) indexer/retriever tests
# ---------------------------------------------------------------------------


class MockEmbedder:
    """Deterministic mock embedder. No model download required."""

    model_name: str = "mock"
    _dim: int = 8

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), self._dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def embed_query(self, query: str) -> np.ndarray:
        rng = np.random.default_rng(123)
        vec = rng.standard_normal((1, self._dim)).astype(np.float32)
        return vec / np.linalg.norm(vec)

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def sample_vqa_sample():
    from radiology_vqa.schema import VQASample

    image = Image.new("RGB", (64, 64), color=(128, 0, 0))
    return VQASample(
        image=image,
        question="Is there a fracture?",
        answer="yes",
        answer_type="closed",
        modality="xray",
        source="vqa_rad",
        sample_id="vqa_rad_train_0",
    )


@pytest.fixture
def sample_slake_sample():
    from radiology_vqa.schema import SLAKESample

    image = Image.new("RGB", (64, 64), color=(0, 128, 0))
    return SLAKESample(
        image=image,
        question="What organ is shown?",
        answer="liver",
        answer_type="open",
        modality="ct",
        source="slake",
        sample_id="slake_train_1",
        location="abdomen",
        content_type="organ",
        triple=["liver", "is_a", "organ"],
        img_name="xmlab0/source.jpg",
    )


@pytest.fixture
def sample_kg_triple():
    from radiology_vqa.schema import KGTriple

    return KGTriple(head="pneumonia", relation="is_a", tail="disease", category="disease")


@pytest.fixture
def tmp_slake_dir(tmp_path):
    # Image directory
    imgs_dir = tmp_path / "imgs" / "xmlab_test"
    imgs_dir.mkdir(parents=True)

    # KG directory
    kg_dir = tmp_path / "KG"
    kg_dir.mkdir()

    # Synthetic JPEG image
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(imgs_dir / "source.jpg", format="JPEG")

    # train.json: 3 English + 1 Chinese row
    train_data = [
        {
            "qid": 1,
            "img_name": "xmlab_test/source.jpg",
            "question": "Is there a fracture?",
            "answer": "yes",
            "answer_type": "CLOSED",
            "q_lang": "en",
            "modality": "X-Ray",
            "location": "chest",
            "content_type": "abnormality",
            "triple": ["fracture", "is_a", "abnormality"],
        },
        {
            "qid": 2,
            "img_name": "xmlab_test/source.jpg",
            "question": "What modality is this?",
            "answer": "X-Ray",
            "answer_type": "OPEN",
            "q_lang": "en",
            "modality": "X-Ray",
            "location": "chest",
            "content_type": "modality",
            "triple": [],
        },
        {
            "qid": 3,
            "img_name": "xmlab_test/source.jpg",
            "question": "Is this an MRI?",
            "answer": "no",
            "answer_type": "CLOSED",
            "q_lang": "en",
            "modality": "X-Ray",
            "location": "",
            "content_type": "modality",
            "triple": [],
        },
        {
            "qid": 4,
            "img_name": "xmlab_test/source.jpg",
            "question": "这是什么器官？",
            "answer": "心脏",
            "answer_type": "OPEN",
            "q_lang": "zh",
            "modality": "X-Ray",
            "location": "",
            "content_type": "organ",
            "triple": [],
        },
    ]
    with open(tmp_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f)

    # KG/en_disease.csv (2 valid rows)
    disease_csv = "head#relation#tail\npneumonia#is_a#disease\nfracture#is_a#abnormality\n"
    (kg_dir / "en_disease.csv").write_text(disease_csv, encoding="utf-8")

    # macOS resource fork decoy — must be ignored
    (kg_dir / "._en_disease.csv").write_text("junk", encoding="utf-8")

    return tmp_path


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


@pytest.fixture
def sample_documents():
    from radiology_vqa.rag.document import Document, DocumentMeta

    entries = [
        ("kg_disease", "Pneumonia", "symptom", "en_disease.csv", "Pneumonia symptoms include: fever, cough, shortness of breath."),
        ("kg_disease", "Tuberculosis", "cause", "en_disease.csv", "Tuberculosis is caused by: Mycobacterium tuberculosis."),
        ("kg_disease", "Asthma", "treatment", "en_disease.csv", "Treatment for Asthma: bronchodilators, corticosteroids."),
        ("kg_organ", "Liver", "function", "en_organ.csv", "The function of Liver: metabolize nutrients, detoxify blood."),
        ("kg_organ", "Lung", "function", "en_organ.csv", "The function of Lung: gas exchange, oxygen intake."),
        ("kg_organ", "Heart", "function", "en_organ.csv", "The function of Heart: pump blood throughout the body."),
        ("kg_organ", "Kidney", "definition", "en_organ.csv", "Kidney: organ that filters blood and produces urine."),
        ("kg_organ_rel", "Liver", "belong_to", "en_organ_rel.csv", "Liver belongs to the Digestive System."),
        ("kg_organ_rel", "Heart", "belong_to", "en_organ_rel.csv", "Heart belongs to the Circulatory System."),
        ("kg_organ_rel", "Brain", "belong_to", "en_organ_rel.csv", "Brain belongs to the Nervous System."),
    ]
    return [
        Document(
            text=text,
            meta=DocumentMeta(source_type=st, entity_name=en, attribute=attr, source_file=sf),
            doc_id=f"{st}_{en.lower()}_{attr}_0",
        )
        for st, en, attr, sf, text in entries
    ]


@pytest.fixture
def sample_kg_triples_mixed():
    from radiology_vqa.schema import KGTriple

    return [
        KGTriple(head="Pneumonia", relation="symptom", tail="fever, cough", category="disease"),
        KGTriple(head="Tuberculosis", relation="cause", tail="Mycobacterium tuberculosis", category="disease"),
        KGTriple(head="Liver", relation="function", tail="metabolize nutrients", category="organ"),
        KGTriple(head="Lung", relation="definition", tail="respiratory organ", category="organ"),
        KGTriple(head="Liver", relation="belong to", tail="Digestive System", category="organ_rel"),
        KGTriple(head="Lung", relation="belong to", tail="Respiratory System", category="organ_rel"),
    ]


@pytest.fixture
def tmp_index_dir(tmp_path):
    return tmp_path / "index"


# ---------------------------------------------------------------------------
# MockVLMBackend — zero-dependency mock for fast benchmark/VLM tests
# ---------------------------------------------------------------------------


class MockVLMBackend:
    """Deterministic mock VLM backend. No model download required.

    Returns a configurable fixed answer for every prediction.
    Satisfies :class:`VLMInterface` via duck typing.
    """

    def __init__(self, fixed_answer: str = "yes") -> None:
        self._fixed_answer = fixed_answer

    def predict(self, image, question):
        from radiology_vqa.vlm.interface import VLMPrediction

        return VLMPrediction(
            answer=self._fixed_answer,
            confidence=0.5,
            raw_output=self._fixed_answer,
            model_name="mock",
            latency_seconds=0.01,
        )

    def predict_batch(self, samples):
        return [self.predict(img, q) for img, q in samples]

    @property
    def model_name(self) -> str:
        return "mock"


@pytest.fixture
def mock_vlm_backend():
    return MockVLMBackend()


@pytest.fixture
def sample_vqa_samples_for_benchmark():
    """10 VQASamples: 5 closed (answer='yes') + 5 open (answer='pneumonia')."""
    from radiology_vqa.schema import VQASample

    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    samples = []

    for i in range(5):
        samples.append(
            VQASample(
                image=img,
                question=f"Is there an abnormality? ({i})",
                answer="yes",
                answer_type="closed",
                modality="xray",
                source="vqa_rad",
                sample_id=f"vqa_rad_test_closed_{i}",
            )
        )

    for i in range(5):
        samples.append(
            VQASample(
                image=img,
                question=f"What is the finding? ({i})",
                answer="pneumonia",
                answer_type="open",
                modality="xray",
                source="vqa_rad",
                sample_id=f"vqa_rad_test_open_{i}",
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Phase 4A agent node fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state():
    """Minimal valid input state with a synthetic gray image."""
    return {
        "image": Image.new("RGB", (224, 224), color="gray"),
        "question": "Is there evidence of pneumonia?",
        "answer_type": "closed",
        "retry_count": 0,
    }


@pytest.fixture
def mock_vlm():
    """Mock VLMInterface returning a high-confidence prediction. Tracks last call."""
    from radiology_vqa.vlm.interface import VLMPrediction

    class MockVLM:
        def __init__(self):
            self.last_image = None
            self.last_question = None

        def predict(self, image, question):
            self.last_image = image
            self.last_question = question
            return VLMPrediction(
                answer="yes",
                confidence=0.92,
                raw_output="yes",
                model_name="mock-vlm",
                latency_seconds=0.01,
            )

        def predict_batch(self, samples):
            return [self.predict(img, q) for img, q in samples]

        @property
        def model_name(self):
            return "mock-vlm"

    return MockVLM()


@pytest.fixture
def mock_retriever():
    """Returns the MockRetriever CLASS (not an instance).

    Usage in tests:
        retriever = mock_retriever()                   # empty results
        retriever = mock_retriever(results=[...])      # with results
    """

    class MockRetriever:
        def __init__(self, results=None):
            self._results = results or []

        def retrieve(self, query, top_k=5, min_score=0.0):
            return self._results

        def retrieve_with_filter(self, query, top_k=5, source_type=None):
            return self._results

    return MockRetriever


@pytest.fixture
def sample_evidence():
    """Sample evidence list in plain-dict format as stored in AgentState.

    All three items relate to Lobar Pneumonia, so they support queries
    that contain the keyword 'pneumonia'.
    """
    return [
        {
            "text": "Lobar Pneumonia symptoms include: chills, high fever, chest pain, cough, rusty sputum",
            "score": 0.72,
            "source_type": "kg_disease",
            "entity_name": "Lobar Pneumonia",
            "attribute": "symptom",
            "rank": 1,
        },
        {
            "text": "Lobar Pneumonia is caused by: infection due to streptococcus pneumoniae",
            "score": 0.65,
            "source_type": "kg_disease",
            "entity_name": "Lobar Pneumonia",
            "attribute": "cause",
            "rank": 2,
        },
        {
            "text": "Treatment for Lobar Pneumonia: antibiotic drug therapy",
            "score": 0.58,
            "source_type": "kg_disease",
            "entity_name": "Lobar Pneumonia",
            "attribute": "treatment",
            "rank": 3,
        },
    ]


# ---------------------------------------------------------------------------
# Phase 4B graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lightweight_graph():
    """Compiled LangGraph using build_lightweight() — no VLM or Retriever needed."""
    from radiology_vqa.config import Settings
    from radiology_vqa.graph.builder import GraphBuilder

    builder = GraphBuilder(Settings())
    return builder.build_lightweight()


@pytest.fixture
def pre_populated_state(base_state, sample_evidence):
    """AgentState with visual agent and retrieval agent outputs already filled in.

    Use this to test the supervisor → output_formatter flow through the
    lightweight graph without needing a real VLM or FAISS index.
    """
    return {
        **base_state,
        "visual_answer": "yes",
        "visual_confidence": 0.92,
        "visual_raw_output": "yes",
        "visual_model": "mock-vlm",
        "visual_error": "",
        "retrieval_query": "Is there evidence of pneumonia? yes",
        "retrieved_evidence": sample_evidence,
        "retrieval_error": "",
    }


@pytest.fixture
def synthetic_dicom(tmp_path):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.StudyDate = "20240101"
    ds.Modality = "CR"

    # PHI fields — these should be stripped by anonymize_metadata
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "M"
    ds.PatientAge = "044Y"
    ds.InstitutionName = "Test Hospital"

    # Windowing
    ds.WindowCenter = 128.0
    ds.WindowWidth = 256.0

    # 64×64 uint16 pixel data
    pixel_array = np.arange(0, 4096, dtype=np.uint16).reshape(64, 64)
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.tobytes()

    dicom_path = tmp_path / "test.dcm"
    pydicom.dcmwrite(str(dicom_path), ds)
    return dicom_path

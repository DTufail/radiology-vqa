"""Entry node — validates and normalises inputs before the pipeline starts."""

import logging

from PIL import Image

from radiology_vqa.agents.state import AgentState

logger = logging.getLogger(__name__)

# Question-initial words that indicate a closed (yes/no) question
_CLOSED_QUESTION_PREFIXES: frozenset[str] = frozenset({
    "is", "are", "does", "do", "was", "were",
    "has", "have", "can", "will", "should",
})

_MAX_DIMENSION: int = 4096


def entry_node(state: AgentState) -> AgentState:
    """Validate and normalise inputs before expensive agent calls.

    Reads from state:
        image, question, answer_type, retry_count, visual_error, retrieval_error

    Writes to state:
        image (converted to RGB / resized if needed)
        question (stripped whitespace)
        answer_type (inferred if not already set)
        retry_count (initialised to 0 if absent)
        visual_error (set on validation failure; empty otherwise)
        retrieval_error (initialised to "" if absent)

    This node NEVER raises.  All validation failures go to visual_error so that
    the supervisor can route to abstain.
    """
    updates: dict = {
        "retry_count": state.get("retry_count", 0),
        "visual_error": state.get("visual_error", ""),
        "retrieval_error": state.get("retrieval_error", ""),
    }

    # ── Validate image ──────────────────────────────────────────────────────
    image = state.get("image")
    if image is None or not isinstance(image, Image.Image):
        updates["visual_error"] = "Invalid or missing image"
        logger.warning("Entry node: invalid or missing image (type=%s)", type(image).__name__)
        return {**state, **updates}

    # Convert to RGB if needed
    if image.mode != "RGB":
        orig_mode = image.mode
        image = image.convert("RGB")
        logger.debug("Entry node: converted image from %s to RGB", orig_mode)

    # Resize if any dimension exceeds the cap
    w, h = image.size
    if w > _MAX_DIMENSION or h > _MAX_DIMENSION:
        scale = _MAX_DIMENSION / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.warning(
            "Entry node: image resized from %dx%d to %dx%d (max_dimension=%d)",
            w, h, new_w, new_h, _MAX_DIMENSION,
        )

    updates["image"] = image

    # ── Validate question ───────────────────────────────────────────────────
    question = state.get("question", "").strip()
    if not question:
        updates["visual_error"] = "Empty question"
        logger.warning("Entry node: empty question after strip")
        return {**state, **updates}

    updates["question"] = question

    # ── Infer answer_type if not already set ────────────────────────────────
    answer_type = state.get("answer_type", "")
    if not answer_type:
        tokens = question.split()
        first_word = tokens[0].lower().rstrip("'?") if tokens else ""
        answer_type = "closed" if first_word in _CLOSED_QUESTION_PREFIXES else "open"
        logger.debug(
            "Entry node: inferred answer_type=%r from question %r",
            answer_type, question[:50],
        )

    updates["answer_type"] = answer_type

    logger.debug(
        "Entry node: validated — question=%.60r answer_type=%r",
        question, answer_type,
    )
    return {**state, **updates}

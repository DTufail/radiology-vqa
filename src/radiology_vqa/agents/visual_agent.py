"""Visual Agent node — runs VLM inference on the image-question pair."""

import logging

from radiology_vqa.agents.state import AgentState
from radiology_vqa.vlm.interface import VLMInterface

logger = logging.getLogger(__name__)


def visual_agent_node(state: AgentState, vlm: VLMInterface) -> AgentState:
    """Run VLM inference on the image-question pair.

    Reads from state:
        image, question

    Writes to state:
        visual_answer, visual_confidence, visual_raw_output, visual_model, visual_error

    The VLM is injected, not created here. This node is stateless.
    Errors are written to state so the supervisor can handle them — never raised.
    """
    question = state.get("question", "")
    image = state.get("image")

    try:
        prediction = vlm.predict(image, question)
        logger.info(
            "Visual agent: q=%.80r → answer=%r confidence=%.3f latency=%.2fs",
            question,
            prediction.answer,
            prediction.confidence,
            prediction.latency_seconds,
        )
        updates: dict = {
            "visual_answer": prediction.answer,
            "visual_confidence": prediction.confidence,
            "visual_raw_output": prediction.raw_output,
            "visual_model": prediction.model_name,
            "visual_error": "",
        }
    except Exception as e:
        logger.error("Visual agent failed: %s", e, exc_info=True)
        updates = {
            "visual_answer": "",
            "visual_confidence": 0.0,
            "visual_raw_output": "",
            "visual_model": "",
            "visual_error": str(e),
        }

    return {**state, **updates}

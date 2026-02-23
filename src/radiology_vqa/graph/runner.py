"""AgentRunner — high-level entry point for the multi-agent pipeline.

This is the single class that Phase 7's API will use:

    runner = create_runner()

    @app.post("/predict")
    async def predict(image: UploadFile, question: str):
        pil_image = Image.open(image.file)
        return runner.run_query(pil_image, question).model_dump()
"""

import logging
import time

from PIL import Image

from radiology_vqa.agents.output_formatter import format_system_output
from radiology_vqa.agents.state import AgentState, SystemOutput
from radiology_vqa.config import Settings

logger = logging.getLogger(__name__)

_FALLBACK_ABSTAIN = SystemOutput(
    answer="ABSTAIN: Pipeline error — unable to produce a reliable answer.",
    confidence=0.0,
    citations=[],
    requires_human_review=True,
    reasoning="Catastrophic pipeline failure.",
    decision="abstain",
    visual_answer="",
    retrieval_query="",
)


class AgentRunner:
    """High-level interface for running the multi-agent pipeline.

    The graph (VLM + Retriever) is loaded once at __init__.
    Subsequent run_query() calls reuse the loaded graph — graph.invoke() is
    cheap; model loading is not.
    """

    def __init__(self, config: Settings | None = None) -> None:
        """Load config and compile the graph (including VLM + Retriever)."""
        if config is None:
            from radiology_vqa.config import settings
            config = settings
        self._config = config

        from radiology_vqa.graph.builder import GraphBuilder
        logger.info("AgentRunner: building graph (loading VLM + Retriever)…")
        t0 = time.perf_counter()
        self._graph = GraphBuilder(self._config).build()
        logger.info("AgentRunner: graph ready (%.1fs)", time.perf_counter() - t0)

    def run_query(
        self,
        image: Image.Image,
        question: str,
        answer_type: str = "",
    ) -> SystemOutput:
        """Run the full multi-agent pipeline on a single image-question pair.

        Args:
            image: PIL Image (any mode — entry_node converts to RGB).
            question: The medical question to answer.
            answer_type: "open", "closed", or "" (auto-infer in entry_node).

        Returns:
            SystemOutput with answer, confidence, citations, reasoning.
            Never raises — catastrophic failures return an abstain output.
        """
        t0 = time.perf_counter()
        initial_state: AgentState = {
            "image": image,
            "question": question,
            "answer_type": answer_type,
            "retry_count": 0,
        }

        try:
            final_state: dict = self._graph.invoke(initial_state)
            result = format_system_output(final_state)
            elapsed = time.perf_counter() - t0
            logger.info(
                "run_query: q=%.60r decision=%s answer=%.40r conf=%.3f latency=%.2fs",
                question,
                final_state.get("decision", "?"),
                result.answer,
                result.confidence,
                elapsed,
            )
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.exception(
                "run_query: catastrophic failure (%.2fs): %s", elapsed, exc
            )
            return SystemOutput(
                answer="ABSTAIN: Pipeline error — unable to produce a reliable answer.",
                confidence=0.0,
                citations=[],
                requires_human_review=True,
                reasoning=f"Pipeline error: {exc}",
                decision="abstain",
                visual_answer="",
                retrieval_query="",
            )

    def run_batch(
        self,
        samples: list[tuple[Image.Image, str]],
        answer_types: list[str] | None = None,
    ) -> list[SystemOutput]:
        """Run the pipeline on multiple samples sequentially.

        Args:
            samples: List of (image, question) tuples.
            answer_types: Optional list of answer types (parallel to samples).
                          If None, all samples use "" (auto-infer).

        Returns:
            List of SystemOutput in the same order as the input samples.
        """
        results: list[SystemOutput] = []
        n = len(samples)
        for i, (image, question) in enumerate(samples):
            atype = answer_types[i] if answer_types else ""
            results.append(self.run_query(image, question, answer_type=atype))
            if (i + 1) % 10 == 0 or (i + 1) == n:
                logger.info("run_batch: %d/%d completed", i + 1, n)
        return results


def create_runner(config: Settings | None = None) -> AgentRunner:
    """Factory function for creating an AgentRunner.

    Convenience wrapper for scripts and the Phase 7 API.
    """
    return AgentRunner(config)

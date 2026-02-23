"""Graph construction and compilation for the multi-agent radiology VQA pipeline."""

import logging
from functools import partial

from langgraph.graph import END, START, StateGraph

from radiology_vqa.agents.output_formatter import output_formatter_node
from radiology_vqa.agents.retrieval_agent import retrieval_agent_node
from radiology_vqa.agents.state import AgentState
from radiology_vqa.agents.supervisor import supervisor_node
from radiology_vqa.agents.visual_agent import visual_agent_node
from radiology_vqa.config import Settings
from radiology_vqa.graph.entry import entry_node
from radiology_vqa.graph.routing import route_after_supervisor

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Constructs and compiles the multi-agent LangGraph.

    Usage (production):
        builder = GraphBuilder(config)
        graph = builder.build()
        result = graph.invoke(initial_state)

    Usage (testing — no GPU/index required):
        graph = builder.build_lightweight()
        result = graph.invoke(pre_populated_state)
    """

    def __init__(self, config: Settings) -> None:
        """Store config.  Model loading happens in build(), not here."""
        self._config = config

    def build(
        self,
        vlm=None,
        retriever=None,
    ):
        """Build and compile the graph with a real VLM and Retriever.

        If vlm / retriever are not provided they are created from config.
        Passing pre-loaded instances avoids reloading models on each call and
        is the recommended pattern for the API and benchmarking scripts.

        Args:
            vlm: An object satisfying VLMInterface, or None to create from config.
            retriever: A Retriever instance, or None to create from config.

        Returns:
            A compiled LangGraph that accepts AgentState dicts.
        """
        if vlm is None:
            from radiology_vqa.vlm.factory import create_vlm_backend
            logger.info("GraphBuilder.build(): creating VLM backend (%s)", self._config.vlm_backend)
            vlm = create_vlm_backend(self._config)

        if retriever is None:
            from radiology_vqa.rag.retriever import Retriever
            logger.info("GraphBuilder.build(): loading Retriever from %s", self._config.index_dir)
            retriever = Retriever(self._config.index_dir)

        visual_fn = partial(visual_agent_node, vlm=vlm)
        retrieval_fn = partial(
            retrieval_agent_node,
            retriever=retriever,
            top_k=self._config.retrieval_top_k,
        )

        return self._compile_graph(visual_fn, retrieval_fn)

    def build_lightweight(self):
        """Build the graph WITHOUT loading VLM or Retriever.

        The lightweight visual and retrieval nodes are passthrough functions that
        forward whatever values are already in the state.  This lets tests drive
        the full graph (entry → supervisor → formatter) by pre-populating state
        without touching the GPU or a FAISS index.
        """
        return self._compile_graph(
            _passthrough_visual_node,
            _passthrough_retrieval_node,
        )

    def _compile_graph(self, visual_fn, retrieval_fn):
        """Build the StateGraph, wire edges, compile and return."""
        graph = StateGraph(AgentState)

        graph.add_node("entry", entry_node)
        graph.add_node("visual_agent", visual_fn)
        graph.add_node("retrieval_agent", retrieval_fn)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("output_formatter", output_formatter_node)

        # Linear edges
        graph.add_edge(START, "entry")
        graph.add_edge("entry", "visual_agent")
        graph.add_edge("visual_agent", "retrieval_agent")
        graph.add_edge("retrieval_agent", "supervisor")

        # Conditional edge: supervisor → output_formatter | retrieval_agent (re-query loop)
        graph.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                "output_formatter": "output_formatter",
                "retrieval_agent": "retrieval_agent",
            },
        )

        graph.add_edge("output_formatter", END)

        compiled = graph.compile()
        logger.info(
            "GraphBuilder: compiled graph — nodes: %s",
            list(compiled.get_graph().nodes.keys()),
        )
        return compiled


# ---------------------------------------------------------------------------
# Lightweight passthrough nodes (no VLM / no Retriever)
# ---------------------------------------------------------------------------


def _passthrough_visual_node(state: AgentState) -> dict:
    """Passthrough: forward existing visual fields unchanged.

    Used by build_lightweight() so tests can drive the graph with
    pre-populated visual fields without a real VLM.
    """
    return {
        "visual_answer": state.get("visual_answer", ""),
        "visual_confidence": state.get("visual_confidence", 0.0),
        "visual_raw_output": state.get("visual_raw_output", ""),
        "visual_model": state.get("visual_model", "lightweight-passthrough"),
        "visual_error": state.get("visual_error", ""),
    }


def _passthrough_retrieval_node(state: AgentState) -> dict:
    """Passthrough: forward existing retrieval fields unchanged.

    Used by build_lightweight() so tests can drive the graph with
    pre-populated evidence without a real FAISS index.
    """
    return {
        "retrieval_query": state.get("retrieval_query", state.get("question", "")),
        "retrieved_evidence": state.get("retrieved_evidence", []),
        "retrieval_error": state.get("retrieval_error", ""),
    }

"""Phase 4A agent nodes — pure Python functions, no LangGraph dependency.

Phase 4B wraps these as LangGraph nodes using functools.partial to inject
the vlm and retriever dependencies:

    from functools import partial
    visual_node = partial(visual_agent_node, vlm=vlm_instance)
    retrieval_node = partial(retrieval_agent_node, retriever=retriever_instance, top_k=5)

Phase 4B graph components live in radiology_vqa.graph:
    from radiology_vqa.graph import AgentRunner, GraphBuilder, create_runner
"""

from radiology_vqa.agents.output_formatter import format_system_output, output_formatter_node
from radiology_vqa.agents.retrieval_agent import retrieval_agent_node
from radiology_vqa.agents.state import AgentState, SystemOutput
from radiology_vqa.agents.supervisor import supervisor_node
from radiology_vqa.agents.visual_agent import visual_agent_node

__all__ = [
    "AgentState",
    "SystemOutput",
    "visual_agent_node",
    "retrieval_agent_node",
    "supervisor_node",
    "output_formatter_node",
    "format_system_output",
]

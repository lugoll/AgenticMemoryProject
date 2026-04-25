from __future__ import annotations

import logging
import uuid
from typing import Any

import litellm
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config.settings import ExperimentConfig
from src.memory.base import BaseMemory

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """LangGraph state carried between graph nodes during a single reasoning turn."""

    run_id: str
    question: str
    retrieved_context: list[str]
    answer: str
    steps_taken: list[str]


class AgentResponse(TypedDict):
    """Final output returned by BaseAgent.run() after the graph completes."""

    run_id: str
    question: str
    answer: str
    context_used: list[str]
    steps_taken: list[str]
    backend_name: str


class BaseAgent:
    """
    Concrete LangGraph agent that is fully functional out of the box.

    Subclasses only need to provide the memory backend (and optionally override
    _SYSTEM_PROMPT or _build_graph for future topology changes like ReAct loops).

    Graph shape:
        START → retrieve_node → reason_node → END

    Every LLM call in _reason_node is routed through LiteLLM and tagged with
    phase=agent_reasoning / actor=langgraph_node so TelemetryTracker captures it.
    Model name and sampling parameters come exclusively from ExperimentConfig —
    nothing is hardcoded.

    Each call to run() generates a unique run_id (8-char hex UUID) that is
    threaded through AgentState into the LiteLLM metadata, allowing telemetry
    records to be correlated with QA results by joining on run_id.

    Extending:
        - Swap memory: subclass and pass a different BaseMemory to super().__init__()
        - Swap prompt: override _SYSTEM_PROMPT at the class level
        - Change topology: override _build_graph() to add conditional edges / tool nodes
    """

    _SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Answer the user's question using only "
        "the provided context. If the context is empty, say so honestly."
    )

    def __init__(self, memory: BaseMemory, config: ExperimentConfig) -> None:
        self._memory = memory
        self._config = config
        self._graph = self._build_graph()

    # ---- Memory tool interface (agent must only use these two) ----

    def _search_memory(self, query: str) -> list[str]:
        """SearchMemory tool: retrieve relevant context from the memory backend."""
        results = self._memory.search(query)
        logger.info(
            "[tool] SearchMemory  query=%r  results=%d  backend=%s",
            query,
            len(results),
            self._memory.get_backend_name(),
        )
        return results

    def _update_memory(self, fact: str) -> None:
        """UpdateMemory tool: persist a new fact to the memory backend."""
        logger.info(
            "[tool] UpdateMemory  fact=%r  backend=%s",
            fact[:120],
            self._memory.get_backend_name(),
        )
        self._memory.update_fact(fact)

    # ---- LangGraph nodes (override to customise behaviour) ----

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """
        LangGraph node: call SearchMemory and return updated state.

        Populates retrieved_context using the question as the query and appends
        "retrieve" to steps_taken. Does not mutate the incoming state dict.
        """
        context = self._search_memory(state["question"])
        logger.debug("%s._retrieve_node: retrieved %d passages", self.get_agent_name(), len(context))
        return {
            **state,
            "retrieved_context": context,
            "steps_taken": state["steps_taken"] + ["retrieve"],
        }

    def _reason_node(self, state: AgentState) -> AgentState:
        """
        LangGraph node: call the configured LLM via LiteLLM and return updated state.

        The call is tagged with phase=agent_reasoning / actor=langgraph_node so
        TelemetryTracker attributes the tokens to the correct phase and actor.
        run_id is included so every telemetry record can be joined with the
        corresponding QA result. Does not mutate the incoming state dict.
        """
        context_str = "\n".join(state["retrieved_context"]) or "No context available."
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {state['question']}",
            },
        ]

        completion_kwargs: dict[str, Any] = dict(
            model=self._config.llm.model,
            messages=messages,
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
            metadata={
                "phase": "agent_reasoning",
                "actor": "langgraph_node",
                "variant_name": self._config.variant_name,
                "run_id": state["run_id"],
            },
        )
        if self._config.llm.api_base:
            completion_kwargs["api_base"] = self._config.llm.api_base
        response = litellm.completion(**completion_kwargs)

        answer: str = response.choices[0].message.content or ""
        logger.debug("%s._reason_node: received %d-char answer", self.get_agent_name(), len(answer))
        return {
            **state,
            "answer": answer,
            "steps_taken": state["steps_taken"] + ["reason"],
        }

    # ---- Graph construction (override to change topology) ----

    def _build_graph(self) -> Any:
        """
        Build and compile the LangGraph StateGraph.

        Default topology: START → retrieve → reason → END.
        Override this method to introduce conditional edges, tool-call loops,
        or additional nodes without touching the node implementations.
        """
        graph: StateGraph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("reason", self._reason_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", END)
        return graph.compile()

    # ---- Public API ----

    def run(self, question: str) -> AgentResponse:
        """
        Execute the full reasoning graph for a single question.

        Generates a unique run_id for this invocation so telemetry records and
        QA results can be correlated by joining on run_id.

        Args:
            question: Natural language question for the agent to answer.

        Returns:
            AgentResponse containing the answer, context used, execution trace,
            the name of the active memory backend, and the run_id.
        """
        run_id = str(uuid.uuid4())
        initial_state: AgentState = {
            "run_id": run_id,
            "question": question,
            "retrieved_context": [],
            "answer": "",
            "steps_taken": [],
        }
        final_state: AgentState = self._graph.invoke(initial_state)
        return AgentResponse(
            run_id=final_state["run_id"],
            question=final_state["question"],
            answer=final_state["answer"],
            context_used=final_state["retrieved_context"],
            steps_taken=final_state["steps_taken"],
            backend_name=self._memory.get_backend_name(),
        )

    def get_agent_name(self) -> str:
        """Returns the string identifier for this agent. Used for telemetry tagging."""
        return self.__class__.__name__.lower()

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.agent_dummy import DummyAgent
from src.agent.base import AgentState, BaseAgent
from src.config.settings import ExperimentConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent(dummy_experiment_config: ExperimentConfig) -> DummyAgent:
    return DummyAgent(config=dummy_experiment_config)


def _mock_completion(content: str = "mocked answer") -> MagicMock:
    """Return a MagicMock shaped like a litellm.completion response."""
    m = MagicMock()
    m.choices[0].message.content = content
    return m


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------

class TestBaseAgentContract:
    def test_dummy_is_subclass_of_base(self) -> None:
        assert issubclass(DummyAgent, BaseAgent)

    def test_base_agent_is_directly_instantiable(
        self, dummy_experiment_config: ExperimentConfig
    ) -> None:
        """BaseAgent is concrete — it can be instantiated with any memory + config."""
        from src.memory.model_dummy import DummyMemory
        agent = BaseAgent(
            memory=DummyMemory(top_k=dummy_experiment_config.retrieval.top_k),
            config=dummy_experiment_config,
        )
        assert agent is not None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestDummyAgentInit:
    def test_agent_name_is_dummy(self, agent: DummyAgent) -> None:
        assert agent.get_agent_name() == "dummy"

    def test_backend_is_dummy(self, agent: DummyAgent) -> None:
        assert agent._memory.get_backend_name() == "dummy"

    def test_top_k_comes_from_config(
        self, dummy_experiment_config: ExperimentConfig
    ) -> None:
        agent = DummyAgent(config=dummy_experiment_config)
        assert agent._memory._top_k == dummy_experiment_config.retrieval.top_k  # type: ignore[attr-defined]

    def test_graph_is_built_on_init(self, agent: DummyAgent) -> None:
        assert agent._graph is not None


# ---------------------------------------------------------------------------
# Memory tool interface
# ---------------------------------------------------------------------------

class TestSearchMemoryTool:
    def test_search_delegates_to_memory(self, agent: DummyAgent) -> None:
        agent._memory.ingest_documents(["Paris is the capital of France"])
        results = agent._search_memory("Paris France")
        assert len(results) == 1
        assert "Paris" in results[0]

    def test_search_empty_store_returns_empty(self, agent: DummyAgent) -> None:
        assert agent._search_memory("anything") == []

    def test_search_returns_list_of_strings(self, agent: DummyAgent) -> None:
        agent._memory.ingest_documents(["hello world"])
        results = agent._search_memory("hello")
        assert all(isinstance(r, str) for r in results)

    def test_search_empty_query_returns_empty(self, agent: DummyAgent) -> None:
        agent._memory.ingest_documents(["something useful"])
        assert agent._search_memory("") == []


class TestUpdateMemoryTool:
    def test_update_delegates_to_memory(self, agent: DummyAgent) -> None:
        agent._update_memory("The sky is blue")
        assert agent._search_memory("sky") != []

    def test_update_empty_fact_ignored(self, agent: DummyAgent) -> None:
        agent._update_memory("")
        assert agent._memory.store_size == 0  # type: ignore[attr-defined]

    def test_update_fact_immediately_searchable(self, agent: DummyAgent) -> None:
        agent._update_memory("Elephants are the largest land animals")
        assert len(agent._search_memory("elephants animals")) == 1


# ---------------------------------------------------------------------------
# _retrieve_node (no LLM — no mock needed)
# ---------------------------------------------------------------------------

class TestRetrieveNode:
    def _blank_state(self, question: str = "test") -> AgentState:
        return {"run_id": "test-run", "question": question, "retrieved_context": [], "answer": "", "steps_taken": []}

    def test_populates_retrieved_context(self, agent: DummyAgent) -> None:
        agent._memory.ingest_documents(["cats are mammals"])
        result = agent._retrieve_node(self._blank_state("cats"))
        assert "cats are mammals" in result["retrieved_context"]

    def test_appends_retrieve_to_steps(self, agent: DummyAgent) -> None:
        result = agent._retrieve_node(self._blank_state())
        assert result["steps_taken"] == ["retrieve"]

    def test_preserves_prior_steps(self, agent: DummyAgent) -> None:
        state: AgentState = {**self._blank_state(), "steps_taken": ["prior"]}  # type: ignore[typeddict-item]
        assert agent._retrieve_node(state)["steps_taken"] == ["prior", "retrieve"]

    def test_does_not_mutate_input_state(self, agent: DummyAgent) -> None:
        original: list[str] = []
        agent._retrieve_node({**self._blank_state(), "steps_taken": original})
        assert original == []

    def test_empty_store_yields_empty_context(self, agent: DummyAgent) -> None:
        assert agent._retrieve_node(self._blank_state())["retrieved_context"] == []


# ---------------------------------------------------------------------------
# _reason_node (calls litellm — mock required)
# ---------------------------------------------------------------------------

class TestReasonNode:
    def _blank_state(
        self,
        question: str = "test",
        context: list[str] | None = None,
        steps: list[str] | None = None,
    ) -> AgentState:
        return {
            "run_id": "test-run",
            "question": question,
            "retrieved_context": context or [],
            "answer": "",
            "steps_taken": steps or [],
        }

    def test_answer_comes_from_llm(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion("LLM says Paris")):
            result = agent._reason_node(self._blank_state())
        assert result["answer"] == "LLM says Paris"

    def test_answer_is_string(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion("some answer")):
            result = agent._reason_node(self._blank_state())
        assert isinstance(result["answer"], str)

    def test_context_is_forwarded_to_llm(self, agent: DummyAgent) -> None:
        state = self._blank_state(context=["Paris is in France"])
        with patch("litellm.completion", return_value=_mock_completion()) as mock_call:
            agent._reason_node(state)
        user_content: str = mock_call.call_args.kwargs["messages"][1]["content"]
        assert "Paris is in France" in user_content

    def test_question_is_forwarded_to_llm(self, agent: DummyAgent) -> None:
        state = self._blank_state(question="Where is Paris?")
        with patch("litellm.completion", return_value=_mock_completion()) as mock_call:
            agent._reason_node(state)
        user_content: str = mock_call.call_args.kwargs["messages"][1]["content"]
        assert "Where is Paris?" in user_content

    def test_litellm_call_uses_config_model(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()) as mock_call:
            agent._reason_node(self._blank_state())
        assert mock_call.call_args.kwargs["model"] == agent._config.llm.model

    def test_litellm_call_carries_telemetry_tags(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()) as mock_call:
            agent._reason_node(self._blank_state())
        metadata = mock_call.call_args.kwargs["metadata"]
        assert metadata["phase"] == "agent_reasoning"
        assert metadata["actor"] == "langgraph_node"

    def test_appends_reason_to_steps(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            result = agent._reason_node(self._blank_state(steps=["retrieve"]))
        assert result["steps_taken"] == ["retrieve", "reason"]

    def test_does_not_mutate_input_state(self, agent: DummyAgent) -> None:
        original: list[str] = ["retrieve"]
        with patch("litellm.completion", return_value=_mock_completion()):
            agent._reason_node(self._blank_state(steps=list(original)))  # type: ignore[arg-type]
        assert original == ["retrieve"]

    def test_empty_content_fallback_is_empty_string(self, agent: DummyAgent) -> None:
        mock = MagicMock()
        mock.choices[0].message.content = None
        with patch("litellm.completion", return_value=mock):
            result = agent._reason_node(self._blank_state())
        assert result["answer"] == ""

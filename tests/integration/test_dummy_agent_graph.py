"""
Integration tests for DummyAgent: full graph execution via agent.run().

litellm.completion is mocked so these tests run in the default suite without
a live LLM endpoint. They verify graph wiring, state flow, and memory
integration — not LLM output quality.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.agent.agent_dummy import DummyAgent
from src.agent.base import AgentResponse
from src.config.settings import ExperimentConfig


def _mock_completion(content: str = "mocked answer") -> MagicMock:
    m = MagicMock()
    m.choices[0].message.content = content
    return m


@pytest.fixture()
def agent(dummy_experiment_config: ExperimentConfig) -> DummyAgent:
    return DummyAgent(config=dummy_experiment_config)


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

class TestAgentResponseShape:
    def test_run_returns_all_required_keys(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("test question")
        for key in ("run_id", "question", "answer", "context_used", "steps_taken", "backend_name"):
            assert key in response

    def test_question_preserved_in_response(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("What is the capital of France?")
        assert response["question"] == "What is the capital of France?"

    def test_backend_name_is_dummy(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("anything")
        assert response["backend_name"] == "dummy"

    def test_answer_is_the_llm_response(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion("specific answer")):
            response = agent.run("anything")
        assert response["answer"] == "specific answer"

    def test_context_used_is_list(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("anything")
        assert isinstance(response["context_used"], list)

    def test_steps_taken_is_list_of_strings(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("anything")
        assert all(isinstance(s, str) for s in response["steps_taken"])


# ---------------------------------------------------------------------------
# Graph execution and node wiring
# ---------------------------------------------------------------------------

class TestAgentGraphExecution:
    def test_steps_are_retrieve_then_reason(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("test")
        assert response["steps_taken"] == ["retrieve", "reason"]

    def test_empty_memory_yields_empty_context(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("anything")
        assert response["context_used"] == []

    def test_seeded_memory_context_appears_in_response(self, agent: DummyAgent) -> None:
        agent._memory.ingest_documents([
            "Paris is the capital of France",
            "Berlin is the capital of Germany",
        ])
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("What is the capital of France?")
        assert any("Paris" in ctx for ctx in response["context_used"])

    def test_retrieved_context_is_forwarded_to_llm(self, agent: DummyAgent) -> None:
        agent._memory.ingest_documents(["The moon orbits the Earth"])
        with patch("litellm.completion", return_value=_mock_completion()) as mock_call:
            agent.run("moon orbit")
        user_content: str = mock_call.call_args.kwargs["messages"][1]["content"]
        assert "moon orbits" in user_content

    def test_top_k_limits_context_used(
        self, dummy_experiment_config: ExperimentConfig
    ) -> None:
        from src.config.settings import ExperimentConfig
        # Build a config with top_k=2
        cfg_dict = dummy_experiment_config.model_dump()
        cfg_dict["retrieval"]["top_k"] = 2
        cfg = ExperimentConfig.model_validate(cfg_dict)
        agent = DummyAgent(config=cfg)
        agent._memory.ingest_documents([f"apple item {i}" for i in range(10)])
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("apple")
        assert len(response["context_used"]) <= 2

    def test_update_memory_then_run_finds_fact(self, agent: DummyAgent) -> None:
        agent._update_memory("Elephants are the largest land animals")
        with patch("litellm.completion", return_value=_mock_completion()):
            response = agent.run("elephants animals")
        assert any("Elephants" in ctx for ctx in response["context_used"])

    def test_multiple_independent_runs(self, agent: DummyAgent) -> None:
        agent._memory.ingest_documents(["cats are mammals"])
        with patch("litellm.completion", return_value=_mock_completion()):
            r1 = agent.run("cats")
            r2 = agent.run("dogs")
        assert r1["question"] != r2["question"]
        assert len(r1["context_used"]) > 0
        assert len(r2["context_used"]) == 0

    def test_each_run_has_unique_run_id(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            r1 = agent.run("first")
            r2 = agent.run("second")
        assert r1["run_id"] != r2["run_id"]

    def test_each_run_triggers_exactly_one_llm_call(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()) as mock_call:
            agent.run("first")
            agent.run("second")
        assert mock_call.call_count == 2

    def test_llm_call_carries_correct_telemetry_tags(self, agent: DummyAgent) -> None:
        with patch("litellm.completion", return_value=_mock_completion()) as mock_call:
            agent.run("test")
        metadata = mock_call.call_args.kwargs["metadata"]
        assert metadata["phase"] == "agent_reasoning"
        assert metadata["actor"] == "langgraph_node"
        assert metadata["variant_name"] == agent._config.variant_name

    def test_response_typed_dict_fields_have_correct_types(
        self, agent: DummyAgent
    ) -> None:
        with patch("litellm.completion", return_value=_mock_completion()):
            response: AgentResponse = agent.run("test")
        assert isinstance(response["run_id"], str)
        assert len(response["run_id"]) == 8
        assert isinstance(response["question"], str)
        assert isinstance(response["answer"], str)
        assert isinstance(response["context_used"], list)
        assert isinstance(response["steps_taken"], list)
        assert isinstance(response["backend_name"], str)

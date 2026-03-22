"""Focused tests for deterministic RL components and pipeline integration."""

import json
import os
import sys
import tempfile

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_q_learning_agent_select_update_and_persist():
    from app.core.rl_agent import QLearningAgent

    with tempfile.TemporaryDirectory() as tmp:
        q_path = os.path.join(tmp, "q_table.json")
        agent = QLearningAgent(
            alpha=0.5,
            gamma=0.0,
            epsilon=0.0,
            seed=7,
            q_table_path=q_path,
        )

        state = "len:short|intent:creation|noise:low"
        actions = ["balanced", "aggressive", "safe"]

        # With all-zero Q values and epsilon=0, first action wins tie-break.
        chosen = agent.select_action(state, actions)
        assert chosen == "balanced"

        new_q = agent.update(state, "balanced", reward=0.8)
        assert new_q == 0.4  # 0 + 0.5 * (0.8 - 0)

        # Ensure persisted JSON exists and can be reloaded.
        assert os.path.exists(q_path)
        reloaded = QLearningAgent(q_table_path=q_path, epsilon=0.0)
        assert reloaded.q_table[state]["balanced"] == 0.4


def test_state_encoder_discrete_key_contains_required_dimensions():
    from app.core.state_encoder import encode_state

    prompt = "Create API docs!!! ASAP!!!"
    intent_dict = {
        "actions": ["create"],
        "objects": ["api", "docs"],
        "constraints": [],
        "modifiers": [],
    }
    state = encode_state(prompt, intent_dict)

    assert "len:" in state
    assert "|intent:" in state
    assert "|noise:" in state
    assert state.count("|") == 2


def test_reward_function_balances_quality_and_penalties():
    from app.core.reward import compute_reward

    good = compute_reward(
        compression_ratio=0.6,
        keyword_retention=0.9,
        semantic_score=0.85,
        spelling_error_rate=0.0,
        ambiguity=0.05,
    )
    poor = compute_reward(
        compression_ratio=0.1,
        keyword_retention=0.3,
        semantic_score=0.2,
        spelling_error_rate=0.4,
        ambiguity=0.6,
    )

    assert 0.0 <= good <= 1.0
    assert 0.0 <= poor <= 1.0
    assert good > poor


def test_pipeline_rl_update_and_qtable_written():
    from app.core.pipeline import OptiPromptPipeline, PipelineConfig

    with tempfile.TemporaryDirectory() as tmp:
        q_path = os.path.join(tmp, "optiprompt_q_table.json")
        pipeline = OptiPromptPipeline(q_table_path=q_path)
        config = PipelineConfig(
            mode="balanced",
            seed=42,
            debug=True,
            rl_enabled=True,
            rl_q_table_path=q_path,
            rl_alpha=0.2,
            rl_gamma=0.0,
            rl_epsilon=0.0,
        )

        result = pipeline.optimize(
            "Please create a compact API summary with examples for onboarding.",
            config,
        )

        assert "metrics" in result
        assert "rl_reward" in result["metrics"]
        assert "rl_q_value" in result["metrics"]
        assert os.path.exists(q_path)

        with open(q_path, "r", encoding="utf-8") as fh:
            table = json.load(fh)

        state = result["debug"]["pipeline_steps"]["rl_state"]
        action = result["debug"]["pipeline_steps"]["rl_action"]
        assert state in table
        assert action in table[state]


if __name__ == "__main__":
    test_q_learning_agent_select_update_and_persist()
    test_state_encoder_discrete_key_contains_required_dimensions()
    test_reward_function_balances_quality_and_penalties()
    test_pipeline_rl_update_and_qtable_written()
    print("RL system tests passed")
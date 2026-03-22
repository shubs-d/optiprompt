"""Lightweight deterministic Q-learning agent for strategy selection.

This module intentionally avoids ML libraries and uses a dictionary-backed
Q-table persisted as JSON.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, Optional


class QLearningAgent:
    """Dictionary-based Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.0,
        epsilon: float = 0.05,
        seed: int = 42,
        q_table_path: Optional[str] = None,
    ) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self.gamma = max(0.0, min(1.0, gamma))
        self.epsilon = max(0.0, min(1.0, epsilon))
        self._rng = random.Random(seed)
        self.q_table_path = q_table_path or ".optiprompt_q_table.json"
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.load()

    def select_action(self, state: str, actions: Iterable[str]) -> str:
        """Select an action using epsilon-greedy policy.

        Tie breaks are deterministic based on action order passed by caller.
        """
        available_actions = [a for a in actions if a]
        if not available_actions:
            raise ValueError("actions must contain at least one action")

        self._ensure_state_actions(state, available_actions)

        if self._rng.random() < self.epsilon:
            return self._rng.choice(available_actions)

        state_values = self.q_table[state]
        max_q = max(state_values.get(a, 0.0) for a in available_actions)
        for action in available_actions:
            if state_values.get(action, 0.0) == max_q:
                return action

        return available_actions[0]

    def update(self, state: str, action: str, reward: float) -> float:
        """Update Q-value for (state, action) and persist the table.

        The interface is intentionally compact and treats this as a contextual
        bandit update while still allowing optional bootstrapping via gamma.
        """
        self._ensure_state_actions(state, [action])

        current_q = self.q_table[state].get(action, 0.0)
        max_future = max(self.q_table[state].values()) if self.q_table[state] else 0.0
        target = reward + (self.gamma * max_future)
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action] = round(new_q, 6)
        self.save()
        return self.q_table[state][action]

    def load(self) -> None:
        """Load persisted Q-table from JSON if it exists."""
        if not self.q_table_path or not os.path.exists(self.q_table_path):
            self.q_table = {}
            return

        try:
            with open(self.q_table_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            self.q_table = {}
            return

        if not isinstance(data, dict):
            self.q_table = {}
            return

        normalized: Dict[str, Dict[str, float]] = {}
        for state, values in data.items():
            if not isinstance(state, str) or not isinstance(values, dict):
                continue
            action_map: Dict[str, float] = {}
            for action, q_value in values.items():
                if isinstance(action, str) and isinstance(q_value, (int, float)):
                    action_map[action] = float(q_value)
            normalized[state] = action_map

        self.q_table = normalized

    def save(self) -> None:
        """Persist Q-table to JSON path."""
        if not self.q_table_path:
            return

        directory = os.path.dirname(self.q_table_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.q_table_path, "w", encoding="utf-8") as fh:
            json.dump(self.q_table, fh, indent=2, sort_keys=True, ensure_ascii=True)

    def _ensure_state_actions(self, state: str, actions: Iterable[str]) -> None:
        if state not in self.q_table:
            self.q_table[state] = {}
        for action in actions:
            self.q_table[state].setdefault(action, 0.0)

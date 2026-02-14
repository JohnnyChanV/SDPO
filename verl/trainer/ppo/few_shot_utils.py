# Copyright 2024 Few-shot Context Distillation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Few-shot example selection and prompt construction utilities for Context Distillation.

This module provides:
  - Loading dev_data, distance matrices, and message-level data
  - Per-query dynamic example selection (Neuron Similarity / Jaccard distance)
  - Teacher prompt construction (system / message placement modes)
  - Student prompt construction (zero-shot)
  - Caching of selected example IDs
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dev_data(path: str) -> list[dict]:
    """Load dev_data.json (candidate example pool)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d dev examples from %s", len(data), path)
    return data


def load_dev_msg_data(path: str) -> list[dict]:
    """Load essay_sampling_reasoning.jsonl (messages with parse_pred / ground_t)."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d dev message records from %s", len(records), path)
    return records


def load_distance_matrix(path: str) -> np.ndarray:
    """Load a Jaccard distance matrix (.npy). Shape = (num_query, num_candidate)."""
    mat = np.load(path)
    logger.info("Loaded distance matrix of shape %s from %s", mat.shape, path)
    return mat


# ---------------------------------------------------------------------------
# Per-query few-shot selection
# ---------------------------------------------------------------------------

def select_few_shot_examples(
    query_idx: int,
    k: int,
    distance_matrix: np.ndarray,
    dev_msg_data: list[dict],
    filter_correct_only: bool = True,
    exclude_self: bool = True,
) -> list[dict]:
    """
    Select k few-shot examples for a single query based on Jaccard distance.

    Args:
        query_idx: Index of the current query in the distance matrix (axis 0).
        k: Number of examples to select.
        distance_matrix: Shape (num_query, num_candidate). Smaller = more similar.
        dev_msg_data: List of candidate dicts (must contain 'messages', 'parse_pred', 'ground_t').
        filter_correct_only: If True, only select candidates where teacher predicted correctly.
        exclude_self: If True and query_idx < num_candidate, set self-distance to inf.

    Returns:
        List of selected example dicts (length <= k).
    """
    dists = distance_matrix[query_idx].copy()

    # Exclude self if the matrix is square (dev x dev, used during training)
    if exclude_self and query_idx < len(dists):
        dists[query_idx] = np.inf

    nearest_ids = np.argsort(dists)  # ascending order (closest first)

    selected: list[dict] = []
    counter = 0
    while len(selected) < k and counter < len(nearest_ids):
        cand_idx = nearest_ids[counter]
        candidate = dev_msg_data[cand_idx]
        counter += 1

        if filter_correct_only:
            pred = candidate.get("parse_pred")
            gt = candidate.get("ground_t")
            if pred is None or gt is None or pred != gt:
                continue

        selected.append(candidate)

    if len(selected) < k:
        logger.warning(
            "Only found %d/%d valid examples for query_idx=%d", len(selected), k, query_idx
        )

    return selected


def build_few_shot_cache(
    num_queries: int,
    k: int,
    distance_matrix: np.ndarray,
    dev_msg_data: list[dict],
    filter_correct_only: bool = True,
    exclude_self: bool = True,
) -> dict[int, list[int]]:
    """
    Pre-compute the selected example indices for all queries.

    Returns:
        Mapping from query_idx -> list of selected candidate indices in dev_msg_data.
    """
    cache: dict[int, list[int]] = {}
    for q in range(num_queries):
        dists = distance_matrix[q].copy()
        if exclude_self and q < len(dists):
            dists[q] = np.inf
        nearest_ids = np.argsort(dists)
        selected_ids: list[int] = []
        cursor = 0
        while len(selected_ids) < k and cursor < len(nearest_ids):
            cand_idx = int(nearest_ids[cursor])
            cursor += 1
            candidate = dev_msg_data[cand_idx]
            if filter_correct_only:
                pred = candidate.get("parse_pred")
                gt = candidate.get("ground_t")
                if pred is None or gt is None or pred != gt:
                    continue
            selected_ids.append(cand_idx)
        cache[q] = selected_ids
    logger.info("Built few-shot cache for %d queries (k=%d)", num_queries, k)
    return cache


def maybe_transpose_distance_matrix(distance_matrix: np.ndarray, num_candidates: int) -> np.ndarray:
    """Ensure distance matrix has shape (num_query, num_candidate)."""
    if distance_matrix.ndim != 2:
        raise ValueError(f"Distance matrix must be 2D, got shape={distance_matrix.shape}")
    if distance_matrix.shape[1] == num_candidates:
        return distance_matrix
    if distance_matrix.shape[0] == num_candidates:
        logger.warning(
            "Distance matrix shape %s appears transposed; auto-transposing to align axis-1 with candidates=%d.",
            distance_matrix.shape,
            num_candidates,
        )
        return distance_matrix.T
    raise ValueError(
        "Distance matrix candidate axis mismatch. "
        f"shape={distance_matrix.shape}, expected axis-1 or axis-0 to equal num_candidates={num_candidates}"
    )


def extract_comment_text_from_user_content(user_content: str) -> str:
    """Extract raw comment text from a user message like 'The Comment is: ...'."""
    if not isinstance(user_content, str):
        return ""
    text = user_content.strip()
    # Handle common prompt forms:
    # - "The Comment is: ..."
    # - "Comment: ..."
    # - plain raw comment text
    pattern = re.compile(r"^\s*(the\s+comment\s+is|comment)\s*:\s*", flags=re.IGNORECASE)
    return pattern.sub("", text, count=1).strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Default system prompt for the essay comment classification task
DEFAULT_SYSTEM_PROMPT = (
    "You are provided with a comment that offers feedback on a piece of writing. "
    "Your task is to determine whether the comment includes an explanation.\n\n"
    "A comment includes an explanation if, in addition to expressing an opinion or "
    'judgment (e.g., "This paragraph is strong"), it justifies that opinion by providing reasons.\n\n'
    "The user will provide you with a comment, and you should directly answer whether "
    "the comment includes an answer."
)

# System prompt template with placeholder for examples
DEFAULT_SYSTEM_PROMPT_WITH_EXAMPLES = (
    "You are provided with a comment that offers feedback on a piece of writing. "
    "Your task is to determine whether the comment includes an explanation.\n\n"
    "A comment includes an explanation if, in addition to expressing an opinion or "
    'judgment (e.g., "This paragraph is strong"), it justifies that opinion by providing reasons.\n\n'
    "{examples}"
    "The user will provide you with a comment, and you should directly answer whether "
    "the comment includes an answer."
)


def _format_example_for_system(ex: dict, idx: int) -> str:
    """Format a single example for insertion into the system prompt."""
    comment = ex.get("Comment", "")
    label = ex.get("label", ex.get("ground_t", 0))
    if isinstance(label, str):
        try:
            label = int(label)
        except (ValueError, TypeError):
            label = 0
    tag = "<a>None</a>" if label == 0 else "<a>Have Explanation</a>"
    return f"Example {idx}: \n\t- Comment: {comment}\n\t{tag}\n\n"


def build_teacher_prompt_system(
    query: dict,
    examples: list[dict],
    sys_prompt_template: Optional[str] = None,
) -> list[dict]:
    """
    Build teacher prompt with few-shot examples placed in the system prompt.

    Args:
        query: Dict with at least a 'Comment' key.
        examples: List of selected few-shot examples.
        sys_prompt_template: Template string with {examples} placeholder.

    Returns:
        List of message dicts: [system, user].
    """
    if sys_prompt_template is None:
        sys_prompt_template = DEFAULT_SYSTEM_PROMPT_WITH_EXAMPLES

    example_str = ""
    for i, ex in enumerate(examples, 1):
        example_str += _format_example_for_system(ex, i)

    messages = [
        {"role": "system", "content": sys_prompt_template.format(examples=example_str)},
        {"role": "user", "content": f"The Comment is: {query['Comment']}"},
    ]
    return messages


def build_teacher_prompt_message(
    query: dict,
    examples: list[dict],
    sys_prompt_template: Optional[str] = None,
) -> list[dict]:
    """
    Build teacher prompt with few-shot examples placed as multi-turn messages.

    Each example contributes a user + assistant turn extracted from its 'messages' field.

    Args:
        query: Dict with at least a 'Comment' key.
        examples: List of selected few-shot example dicts (must have 'messages' field).
        sys_prompt_template: System prompt (without {examples} placeholder).

    Returns:
        List of message dicts: [system, ex1_user, ex1_assistant, ..., exN_user, exN_assistant, query_user].
    """
    if sys_prompt_template is None:
        sys_prompt_template = DEFAULT_SYSTEM_PROMPT

    few_shot_messages: list[dict] = []
    for ex in examples:
        msgs = ex.get("messages", [])
        # Take the last 2 messages (user + assistant) from each example
        if len(msgs) >= 2:
            few_shot_messages.extend(msgs[-2:])

    messages = [
        {"role": "system", "content": sys_prompt_template},
    ] + few_shot_messages + [
        {"role": "user", "content": f"The Comment is: {query['Comment']}"},
    ]
    return messages


def build_student_prompt(
    query: dict,
    sys_prompt_template: Optional[str] = None,
) -> list[dict]:
    """
    Build student prompt (zero-shot, no few-shot examples).

    Args:
        query: Dict with at least a 'Comment' key.
        sys_prompt_template: System prompt string.

    Returns:
        List of message dicts: [system, user].
    """
    if sys_prompt_template is None:
        sys_prompt_template = DEFAULT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": sys_prompt_template},
        {"role": "user", "content": f"The Comment is: {query['Comment']}"},
    ]
    return messages


# ---------------------------------------------------------------------------
# FewShotManager: stateful manager loaded once per trainer
# ---------------------------------------------------------------------------

class FewShotManager:
    """
    Manages few-shot data, distance matrices, and example caching for the trainer.

    Initialized once at the start of training. Provides methods to build teacher
    and student prompts for any query in the batch.
    """

    def __init__(
        self,
        dev_data_path: str,
        dev_msg_data_path: str,
        train_distance_matrix_path: Optional[str] = None,
        eval_distance_matrix_path: Optional[str] = None,
        k: int = 30,
        placement: str = "message",
        filter_correct_only: bool = True,
        sys_prompt_template: Optional[str] = None,
    ):
        self.k = k
        self.placement = placement
        self.filter_correct_only = filter_correct_only
        self.sys_prompt_template = sys_prompt_template

        # Load data
        self.dev_data = load_dev_data(dev_data_path)
        self.dev_msg_data = load_dev_msg_data(dev_msg_data_path)

        # Load distance matrices
        self.train_dist_matrix: Optional[np.ndarray] = None
        self.eval_dist_matrix: Optional[np.ndarray] = None

        if train_distance_matrix_path and Path(train_distance_matrix_path).exists():
            self.train_dist_matrix = maybe_transpose_distance_matrix(
                load_distance_matrix(train_distance_matrix_path), num_candidates=len(self.dev_msg_data)
            )
        if eval_distance_matrix_path and Path(eval_distance_matrix_path).exists():
            self.eval_dist_matrix = maybe_transpose_distance_matrix(
                load_distance_matrix(eval_distance_matrix_path), num_candidates=len(self.dev_msg_data)
            )

        # Pre-build caches
        self._train_cache: Optional[dict[int, list[int]]] = None
        self._eval_cache: Optional[dict[int, list[int]]] = None

    def _get_or_build_cache(self, mode: str = "train") -> dict[int, list[int]]:
        if mode == "train":
            if self._train_cache is None and self.train_dist_matrix is not None:
                num_q = self.train_dist_matrix.shape[0]
                self._train_cache = build_few_shot_cache(
                    num_queries=num_q,
                    k=self.k,
                    distance_matrix=self.train_dist_matrix,
                    dev_msg_data=self.dev_msg_data,
                    filter_correct_only=self.filter_correct_only,
                    exclude_self=True,
                )
            return self._train_cache or {}
        else:
            if self._eval_cache is None and self.eval_dist_matrix is not None:
                num_q = self.eval_dist_matrix.shape[0]
                self._eval_cache = build_few_shot_cache(
                    num_queries=num_q,
                    k=self.k,
                    distance_matrix=self.eval_dist_matrix,
                    dev_msg_data=self.dev_msg_data,
                    filter_correct_only=self.filter_correct_only,
                    exclude_self=False,  # test queries != dev candidates
                )
            return self._eval_cache or {}

    def get_teacher_messages(
        self,
        query: dict,
        query_idx: int,
        mode: str = "train",
    ) -> list[dict]:
        """
        Build teacher messages for a single query (with few-shot examples).

        Args:
            query: Dict with 'Comment' key.
            query_idx: Index in the distance matrix.
            mode: "train" or "eval".

        Returns:
            List of message dicts for the teacher prompt.
        """
        cache = self._get_or_build_cache(mode)
        example_ids = cache.get(query_idx, [])
        examples = [self.dev_msg_data[eid] for eid in example_ids]

        if self.placement == "system":
            return build_teacher_prompt_system(
                query, examples,
                sys_prompt_template=self.sys_prompt_template or DEFAULT_SYSTEM_PROMPT_WITH_EXAMPLES,
            )
        else:  # "message"
            return build_teacher_prompt_message(
                query, examples,
                sys_prompt_template=self.sys_prompt_template or DEFAULT_SYSTEM_PROMPT,
            )

    def get_student_messages(self, query: dict) -> list[dict]:
        """Build student messages for a single query (zero-shot)."""
        return build_student_prompt(
            query,
            sys_prompt_template=self.sys_prompt_template or DEFAULT_SYSTEM_PROMPT,
        )

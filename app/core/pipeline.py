"""Deterministic 11-stage optimization pipeline for OptiPrompt.

Stages:
  1.  Regex cleaning
  2.  Tokenization
  3.  Keyword extraction
  4.  Rule-based pruning (filler/redundancy removal)
  5.  Intent graph extraction          (NEW)
  6.  Semantic deduplication            (NEW)
  7.  Clause compression
  8.  Multi-candidate generation
  9.  Self-consistency selection        (NEW)
  10. Textual critic refinement         (NEW)
  11. Evaluation + final selection
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, TypedDict

from app.core.cleaner import clean, normalize_text
from app.core.compressor import compress
from app.core.critic import refine as critic_refine
from app.core.evaluator import evaluate_candidate, fitness_score
from app.core.evolutionary import FITNESS_WEIGHTS, evolve
from app.core.intent_graph import extract_full_intent
from app.core.keyword_extractor import extract_keywords
from app.core.rules import RuleConfig, apply_rules
from app.core.semantic import semantic_compress, SEMANTIC_MODE_CONFIGS
from app.core.semantic_rules import semantic_deduplicate
from app.core.tokenizer import tokenize
from app.core.transformations import (
    generate_candidates,
    compute_agreement_scores,
    select_by_consensus,
)
from app.core.spellcheck import spell_check_text
from app.metrics.cost import estimate_cost_savings
from app.metrics.density import information_density
from app.metrics.retention import instruction_integrity, keyword_retention

Mode = Literal["aggressive", "balanced", "safe"]


class CandidateResult(TypedDict):
    name: str
    text: str
    fitness: float
    metrics: Dict[str, float]


@dataclass(frozen=True)
class PipelineConfig:
    mode: Mode = "balanced"
    seed: int = 42
    include_candidates: bool = False
    debug: bool = False


_MODE_RULE_PROFILES: Dict[Mode, RuleConfig] = {
    "aggressive": RuleConfig(
        remove_fillers=True,
        compress_phrases=True,
        drop_adjectives=True,
        preserve_keywords=True,
        adjective_drop_strength=1.0,
        phrase_compress_strength=1.0,
    ),
    "balanced": RuleConfig(
        remove_fillers=True,
        compress_phrases=True,
        drop_adjectives=True,
        preserve_keywords=True,
        adjective_drop_strength=0.65,
        phrase_compress_strength=0.8,
    ),
    "safe": RuleConfig(
        remove_fillers=True,
        compress_phrases=True,
        drop_adjectives=False,
        preserve_keywords=True,
        adjective_drop_strength=0.2,
        phrase_compress_strength=0.45,
    ),
}

_MODE_COMPRESSION_LEVEL: Dict[Mode, float] = {
    "aggressive": 0.9,
    "balanced": 0.6,
    "safe": 0.25,
}

_MODE_FILLER_STRENGTH: Dict[Mode, float] = {
    "aggressive": 1.0,
    "balanced": 0.8,
    "safe": 0.45,
}

# Semantic deduplication aggressiveness per mode
_MODE_DEDUP_AGGRESSIVENESS: Dict[Mode, float] = {
    "aggressive": 0.8,
    "balanced": 0.5,
    "safe": 0.25,
}


class OptiPromptPipeline:
    """Production-facing deterministic prompt optimization pipeline."""

    def optimize(self, prompt: str, config: PipelineConfig) -> Dict:
        original_prompt = prompt.strip()
        if not original_prompt:
            raise ValueError("Prompt must not be empty.")

        # ── Stage 0: Pre-optimization normalization ──────────────────────
        # Lowercase, fix elongation, normalize slang, remove greetings/fillers
        original_prompt = normalize_text(original_prompt)

        mode = config.mode
        weights = FITNESS_WEIGHTS[mode]

        # ── Stage 1: Regex cleaning ──────────────────────────────────────
        cleaned = clean(original_prompt, filler_strength=_MODE_FILLER_STRENGTH[mode])

        # ── Stage 2: Tokenization ────────────────────────────────────────
        tokens = tokenize(cleaned)

        # ── Stage 3: Keyword extraction ──────────────────────────────────
        keywords, instructions = extract_keywords(tokens)

        # ── Stage 4: Rule-based pruning (filler/redundancy removal) ──────
        stage4_text = apply_rules(cleaned, keywords, _MODE_RULE_PROFILES[mode])

        # ── Stage 5: Intent graph extraction (NEW) ───────────────────────
        full_intent = extract_full_intent(stage4_text, keywords)
        intent_dict = full_intent.to_dict()

        # Apply semantic compression (synonym collapse, redundancy, pruning)
        sem_config = SEMANTIC_MODE_CONFIGS.get(mode, SEMANTIC_MODE_CONFIGS["balanced"])
        stage5_text, semantic_metrics = semantic_compress(
            stage4_text, keywords, mode=mode, config=sem_config,
        )

        # ── Stage 6: Semantic deduplication (NEW) ────────────────────────
        dedup_aggressiveness = _MODE_DEDUP_AGGRESSIVENESS[mode]
        stage6_text, dedup_metrics = semantic_deduplicate(
            stage5_text, keywords, aggressiveness=dedup_aggressiveness,
        )

        # ── Stage 7: Clause compression ──────────────────────────────────
        stage7_text = compress(stage6_text, level=_MODE_COMPRESSION_LEVEL[mode])

        # ── Stage 8: Multi-candidate generation ─────────────────────────
        generated_candidates = generate_candidates(cleaned, keywords, instructions)

        # ── Stage 9: Self-consistency selection (NEW) ────────────────────
        # Compute agreement scores across all generated candidates
        agreement_scores = compute_agreement_scores(generated_candidates)
        # Average agreement across all candidates
        avg_agreement = (
            sum(agreement_scores) / len(agreement_scores)
            if agreement_scores else 0.0
        )

        # Evolutionary optimization (Stage 8b)
        seeded_genomes = [item["genome"] for item in generated_candidates]
        evo_text, evo_genome, evo_metrics, evo_trace = evolve(
            original_text=original_prompt,
            cleaned_text=cleaned,
            keywords=keywords,
            instructions=instructions,
            mode=mode,
            seed=config.seed,
            debug=config.debug,
            initial_genomes=seeded_genomes,
        )

        # ── Stage 10: Textual critic refinement (NEW) ───────────────────
        # Refine the evolutionary best and pipeline compressed outputs
        evo_refined, evo_critic_report = critic_refine(
            evo_text, original_prompt, keywords, instructions,
        )
        pipeline_refined, pipe_critic_report = critic_refine(
            stage7_text, original_prompt, keywords, instructions,
        )

        # ── Stage 11: Evaluation + final selection ──────────────────────
        scored_candidates = self._evaluate_candidates(
            original_prompt,
            keywords,
            instructions,
            weights,
            generated_candidates,
            pipeline_refined,
            evo_refined,
            avg_agreement,
        )

        best = self._select_best(scored_candidates)
        optimized_prompt = best["text"]

        # ── Post-optimization spell correction ───────────────────────────
        optimized_prompt = spell_check_text(optimized_prompt)

        # ── Compute output metrics ──────────────────────────────────────
        original_token_count = len(tokenize(original_prompt))
        optimized_token_count = len(tokenize(optimized_prompt))
        compression_ratio = (
            0.0
            if original_token_count == 0
            else max(0.0, 1.0 - (optimized_token_count / original_token_count))
        )

        kr = keyword_retention(optimized_prompt, keywords)
        ii = instruction_integrity(optimized_prompt, instructions)

        # Semantic confidence: weighted combination of retention metrics
        semantic_confidence = round(
            0.35 * kr
            + 0.25 * semantic_metrics.concepts_preserved
            + 0.25 * ii
            + 0.15 * avg_agreement,
            4,
        )

        metrics = {
            "compression_ratio": round(compression_ratio, 4),
            "token_reduction_percent": round(compression_ratio * 100.0, 2),
            "keyword_retention": kr,
            "instruction_integrity": ii,
            "information_density": information_density(tokenize(optimized_prompt)),
            "estimated_cost_savings": estimate_cost_savings(original_prompt, optimized_prompt, model="gpt-4"),
            "estimated_cost_savings_gpt35": estimate_cost_savings(original_prompt, optimized_prompt, model="gpt-3.5"),
            "original_token_count": original_token_count,
            "optimized_token_count": optimized_token_count,
            "redundancy_removed": round(
                semantic_metrics.redundancy_removed
                + dedup_metrics.total_redundancy_score,
                4,
            ),
            "concepts_preserved": semantic_metrics.concepts_preserved,
            "agreement_score": round(avg_agreement, 4),
            "semantic_confidence_score": semantic_confidence,
        }

        result: Dict = {
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "compression_ratio": metrics["compression_ratio"],
            "token_reduction_percent": metrics["token_reduction_percent"],
            "keyword_retention": metrics["keyword_retention"],
            "information_density": metrics["information_density"],
            "estimated_cost_savings": metrics["estimated_cost_savings"],
            "redundancy_removed": metrics["redundancy_removed"],
            "concepts_preserved": metrics["concepts_preserved"],
            "agreement_score": metrics["agreement_score"],
            "semantic_confidence_score": metrics["semantic_confidence_score"],
            "metrics": metrics,
            "mode": mode,
        }

        if config.include_candidates:
            result["candidates"] = scored_candidates

        if config.debug:
            result["debug"] = {
                "pipeline_steps": {
                    "stage1_cleaned": cleaned,
                    "stage2_tokens": tokens,
                    "stage3_keywords": sorted(keywords),
                    "stage3_instructions": sorted(instructions),
                    "stage4_rule_output": stage4_text,
                    "stage5_intent_graph": intent_dict,
                    "stage5_semantic_output": stage5_text,
                    "stage5_semantic_metrics": semantic_metrics.to_dict(),
                    "stage6_dedup_output": stage6_text,
                    "stage6_dedup_metrics": dedup_metrics.to_dict(),
                    "stage7_compressed": stage7_text,
                    "stage9_agreement_scores": agreement_scores,
                },
                "evolution": {
                    "best_genome": evo_genome.to_dict(),
                    "best_metrics": evo_metrics,
                    "trace": evo_trace,
                },
                "critic": {
                    "evolutionary_report": evo_critic_report.to_dict(),
                    "pipeline_report": pipe_critic_report.to_dict(),
                },
            }

        return result

    @staticmethod
    def _evaluate_candidates(
        original_prompt: str,
        keywords: Set[str],
        instructions: Set[str],
        weights: Dict[str, float],
        generated_candidates: List[Dict],
        stage7_text: str,
        evo_text: str,
        agreement_score: float = 0.0,
    ) -> List[CandidateResult]:
        variants: List[Dict[str, str]] = [
            {"name": candidate["name"], "text": candidate["text"]}
            for candidate in generated_candidates
        ]
        variants.append({"name": "pipeline_compressed", "text": stage7_text})
        variants.append({"name": "evolutionary_best", "text": evo_text})

        scored: List[CandidateResult] = []
        for variant in variants:
            metrics = evaluate_candidate(
                original_prompt,
                variant["text"],
                keywords,
                instructions,
                agreement_score_override=agreement_score,
            )
            fit = fitness_score(metrics, **weights)
            scored.append(
                {
                    "name": variant["name"],
                    "text": variant["text"],
                    "fitness": round(fit, 4),
                    "metrics": metrics,
                }
            )

        return scored

    @staticmethod
    def _select_best(candidates: List[CandidateResult]) -> CandidateResult:
        """
        Deterministic tie-breakers:
        1) Higher fitness
        2) Fewer tokens
        3) Lexicographically smaller text
        """
        return sorted(
            candidates,
            key=lambda c: (-c["fitness"], len(tokenize(c["text"])), c["text"]),
        )[0]

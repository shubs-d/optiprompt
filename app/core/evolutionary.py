"""
evolutionary.py — Stage 8: Evolutionary Optimization (Genetic Algorithm).

Implements a seeded genetic algorithm that evolves prompt transformation
genomes to maximize a weighted fitness function. Fully deterministic
when the same seed is used.

Updated to use 6-term fitness weights (alpha–zeta) matching the new
semantic compression evaluator formula.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

from app.core.genome import Genome, crossover, mutate, random_genome
from app.core.transformations import apply_genome
from app.core.evaluator import evaluate_candidate, fitness_score


# ── Fitness weight presets (7-term: α–η) ─────────────────────────────────────
#
# fitness = α·compression + β·keyword_retention + γ·instruction_integrity
#         + δ·information_density + η·agreement
#         − ε·redundancy_score − ζ·ambiguity_penalty

FITNESS_WEIGHTS = {
    "aggressive": {
        "alpha": 0.35, "beta": 0.15, "gamma": 0.15,
        "delta": 0.10, "epsilon": 0.05, "zeta": 0.05, "eta": 0.15,
    },
    "balanced": {
        "alpha": 0.25, "beta": 0.20, "gamma": 0.20,
        "delta": 0.10, "epsilon": 0.05, "zeta": 0.05, "eta": 0.15,
    },
    "safe": {
        "alpha": 0.10, "beta": 0.25, "gamma": 0.25,
        "delta": 0.10, "epsilon": 0.10, "zeta": 0.05, "eta": 0.15,
    },
}


def evolve(
    original_text: str,
    cleaned_text: str,
    keywords: Set[str],
    instructions: Set[str],
    mode: str = "balanced",
    population_size: int = 12,
    generations: int = 8,
    seed: int = 42,
    debug: bool = False,
    initial_genomes: Optional[List[Genome]] = None,
) -> Tuple[str, Genome, Dict[str, float], Optional[List[Dict]]]:
    """
    Run the evolutionary optimization loop.

    Args:
        original_text: Original prompt (for evaluation).
        cleaned_text: Pre-processed text (post stages 1-5).
        keywords: Extracted keywords.
        instructions: Extracted instruction verbs.
        mode: 'aggressive', 'balanced', or 'safe'.
        population_size: Number of genomes per generation.
        generations: Number of evolution iterations.
        seed: Random seed for determinism.
        debug: If True, return generation-by-generation trace.

    Returns:
        Tuple of:
        - best optimized text
        - best genome
        - best metrics dict
        - debug trace (list of generation info) or None
    """
    rng = random.Random(seed)
    weights = FITNESS_WEIGHTS.get(mode, FITNESS_WEIGHTS["balanced"])
    trace: List[Dict] = [] if debug else None

    # ── Initialize population ────────────────────────────────────────────
    population = _initialize_population(
        population_size,
        mode,
        rng,
        initial_genomes=initial_genomes,
    )

    best_text = cleaned_text
    best_genome = population[0]
    best_metrics: Dict[str, float] = {}
    best_fitness = -1.0

    # ── Evolution loop ───────────────────────────────────────────────────
    for gen in range(generations):
        gen_results: List[Tuple[Genome, str, Dict[str, float], float]] = []

        for genome in population:
            # Apply genome to produce candidate text
            candidate = apply_genome(cleaned_text, keywords, genome)

            # Evaluate candidate
            metrics = evaluate_candidate(
                original_text, candidate, keywords, instructions,
            )
            fit = fitness_score(metrics, **weights)
            genome.fitness = fit

            gen_results.append((genome, candidate, metrics, fit))

        # Sort by fitness (descending)
        gen_results.sort(key=lambda x: x[3], reverse=True)

        # Track global best
        top_genome, top_text, top_metrics, top_fit = gen_results[0]
        if top_fit > best_fitness:
            best_fitness = top_fit
            best_text = top_text
            best_genome = top_genome
            best_metrics = top_metrics

        # Debug trace
        if trace is not None:
            trace.append({
                "generation": gen,
                "best_fitness": round(top_fit, 4),
                "avg_fitness": round(
                    sum(r[3] for r in gen_results) / len(gen_results), 4
                ),
                "best_genome": top_genome.to_dict(),
                "best_compression": top_metrics.get("compression_ratio", 0),
            })

        # ── Selection + reproduction ─────────────────────────────────────
        # Elitism: keep top 25%
        elite_count = max(2, population_size // 4)
        elites = [r[0] for r in gen_results[:elite_count]]

        # Fill rest with crossover + mutation
        new_population: List[Genome] = list(elites)

        while len(new_population) < population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng, rate=0.3)
            new_population.append(child)

        population = new_population

    return best_text, best_genome, best_metrics, trace


def _initialize_population(
    size: int,
    mode: str,
    rng: random.Random,
    initial_genomes: Optional[List[Genome]] = None,
) -> List[Genome]:
    """
    Initialize population with a mix of:
    - Mode-appropriate preset genome
    - Random genomes biased toward the mode
    """
    from app.core.genome import AGGRESSIVE_GENOME, BALANCED_GENOME, SAFE_GENOME

    presets = {
        "aggressive": AGGRESSIVE_GENOME,
        "balanced": BALANCED_GENOME,
        "safe": SAFE_GENOME,
    }

    population: List[Genome] = []

    # Always include the preset for this mode
    preset = presets.get(mode, BALANCED_GENOME)
    population.append(Genome(
        filler_removal_strength=preset.filler_removal_strength,
        compression_level=preset.compression_level,
        keyword_preservation_bias=preset.keyword_preservation_bias,
        structure_weight=preset.structure_weight,
        redundancy_threshold=preset.redundancy_threshold,
        modifier_pruning_level=preset.modifier_pruning_level,
    ))

    # Include externally seeded genomes (for example, stage-6 variants).
    for seeded in initial_genomes or []:
        if len(population) >= size:
            break
        population.append(Genome(
            filler_removal_strength=seeded.filler_removal_strength,
            compression_level=seeded.compression_level,
            keyword_preservation_bias=seeded.keyword_preservation_bias,
            structure_weight=seeded.structure_weight,
            redundancy_threshold=seeded.redundancy_threshold,
            modifier_pruning_level=seeded.modifier_pruning_level,
        ))

    # Fill remaining with mode-biased random genomes
    while len(population) < size:
        g = random_genome(rng)
        # Bias toward mode
        if mode == "aggressive":
            g.compression_level = max(g.compression_level, 0.5)
            g.filler_removal_strength = max(g.filler_removal_strength, 0.6)
            g.redundancy_threshold = max(g.redundancy_threshold, 0.6)
            g.modifier_pruning_level = max(g.modifier_pruning_level, 0.6)
        elif mode == "safe":
            g.keyword_preservation_bias = max(g.keyword_preservation_bias, 0.7)
            g.structure_weight = max(g.structure_weight, 0.5)
            g.redundancy_threshold = min(g.redundancy_threshold, 0.4)
            g.modifier_pruning_level = min(g.modifier_pruning_level, 0.4)
        population.append(g)

    return population

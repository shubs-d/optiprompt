"""
genome.py — Genome Representation for Evolutionary Optimization.

Defines the genome dataclass used by the genetic algorithm,
with mutation, crossover, and seeded random initialization.

Genome has 6 genes:
- filler_removal_strength: how aggressively fillers are removed (0.0–1.0)
- compression_level: clause compression intensity (0.0–1.0)
- keyword_preservation_bias: weight for keeping keywords (0.0–1.0)
- structure_weight: how much original structure is preserved (0.0–1.0)
- redundancy_threshold: aggressiveness of synonym cluster collapsing (0.0–1.0)
- modifier_pruning_level: how aggressively low-value modifiers are pruned (0.0–1.0)
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Genome:
    """
    Genome for the evolutionary prompt optimizer.

    Each gene controls a transformation parameter:
    - filler_removal_strength: how aggressively fillers are removed (0.0–1.0)
    - compression_level: clause compression intensity (0.0–1.0)
    - keyword_preservation_bias: weight for keeping keywords (0.0–1.0)
    - structure_weight: how much original structure is preserved (0.0–1.0)
    - redundancy_threshold: synonym cluster collapse aggressiveness (0.0–1.0)
    - modifier_pruning_level: low-value modifier removal intensity (0.0–1.0)
    """
    filler_removal_strength: float = 0.7
    compression_level: float = 0.5
    keyword_preservation_bias: float = 0.8
    structure_weight: float = 0.5
    redundancy_threshold: float = 0.5
    modifier_pruning_level: float = 0.5
    fitness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "filler_removal_strength": self.filler_removal_strength,
            "compression_level": self.compression_level,
            "keyword_preservation_bias": self.keyword_preservation_bias,
            "structure_weight": self.structure_weight,
            "redundancy_threshold": self.redundancy_threshold,
            "modifier_pruning_level": self.modifier_pruning_level,
            "fitness": self.fitness,
        }

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        return Genome(
            filler_removal_strength=d.get("filler_removal_strength", 0.7),
            compression_level=d.get("compression_level", 0.5),
            keyword_preservation_bias=d.get("keyword_preservation_bias", 0.8),
            structure_weight=d.get("structure_weight", 0.5),
            redundancy_threshold=d.get("redundancy_threshold", 0.5),
            modifier_pruning_level=d.get("modifier_pruning_level", 0.5),
            fitness=d.get("fitness", 0.0),
        )


def random_genome(rng: random.Random) -> Genome:
    """Create a random genome using a seeded RNG."""
    return Genome(
        filler_removal_strength=round(rng.uniform(0.3, 1.0), 3),
        compression_level=round(rng.uniform(0.1, 1.0), 3),
        keyword_preservation_bias=round(rng.uniform(0.5, 1.0), 3),
        structure_weight=round(rng.uniform(0.1, 0.9), 3),
        redundancy_threshold=round(rng.uniform(0.2, 0.9), 3),
        modifier_pruning_level=round(rng.uniform(0.2, 0.9), 3),
    )


def mutate(genome: Genome, rng: random.Random, rate: float = 0.3) -> Genome:
    """
    Mutate a genome by perturbing its genes with probability `rate`.
    Returns a new Genome (does not modify in-place).
    """
    def _perturb(value: float) -> float:
        if rng.random() < rate:
            delta = rng.gauss(0, 0.15)
            return round(max(0.0, min(1.0, value + delta)), 3)
        return value

    return Genome(
        filler_removal_strength=_perturb(genome.filler_removal_strength),
        compression_level=_perturb(genome.compression_level),
        keyword_preservation_bias=_perturb(genome.keyword_preservation_bias),
        structure_weight=_perturb(genome.structure_weight),
        redundancy_threshold=_perturb(genome.redundancy_threshold),
        modifier_pruning_level=_perturb(genome.modifier_pruning_level),
    )


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
) -> Genome:
    """
    Single-point crossover between two parent genomes.
    Each gene is randomly picked from one of the two parents.
    """
    return Genome(
        filler_removal_strength=(
            parent_a.filler_removal_strength if rng.random() < 0.5
            else parent_b.filler_removal_strength
        ),
        compression_level=(
            parent_a.compression_level if rng.random() < 0.5
            else parent_b.compression_level
        ),
        keyword_preservation_bias=(
            parent_a.keyword_preservation_bias if rng.random() < 0.5
            else parent_b.keyword_preservation_bias
        ),
        structure_weight=(
            parent_a.structure_weight if rng.random() < 0.5
            else parent_b.structure_weight
        ),
        redundancy_threshold=(
            parent_a.redundancy_threshold if rng.random() < 0.5
            else parent_b.redundancy_threshold
        ),
        modifier_pruning_level=(
            parent_a.modifier_pruning_level if rng.random() < 0.5
            else parent_b.modifier_pruning_level
        ),
    )


# ── Preset genomes for mode-based candidate generation ───────────────────────

AGGRESSIVE_GENOME = Genome(
    filler_removal_strength=1.0,
    compression_level=0.95,
    keyword_preservation_bias=0.6,
    structure_weight=0.2,
    redundancy_threshold=0.85,
    modifier_pruning_level=0.9,
)

BALANCED_GENOME = Genome(
    filler_removal_strength=0.7,
    compression_level=0.5,
    keyword_preservation_bias=0.85,
    structure_weight=0.5,
    redundancy_threshold=0.5,
    modifier_pruning_level=0.55,
)

SAFE_GENOME = Genome(
    filler_removal_strength=0.4,
    compression_level=0.2,
    keyword_preservation_bias=0.95,
    structure_weight=0.8,
    redundancy_threshold=0.25,
    modifier_pruning_level=0.2,
)

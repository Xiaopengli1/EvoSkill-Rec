from __future__ import annotations


class EvolutionError(Exception):
    """Base error for Stage 2 evolution components."""


class GenomeValidationError(EvolutionError):
    """Raised when a skill genome is structurally or semantically invalid."""


class MutationValidationError(EvolutionError):
    """Raised when a mutation plan cannot be safely applied."""


class ProposalValidationError(EvolutionError):
    """Raised when an open-ended code proposal fails validation."""


class StaticCodeError(ProposalValidationError):
    """Raised when generated code fails static safety checks."""

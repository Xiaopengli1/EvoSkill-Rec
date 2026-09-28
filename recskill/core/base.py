from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .context import SkillContext
from .spec import SkillSpec


class BaseSkill(torch.nn.Module, ABC):
    """Base class for PyTorch recommendation skills."""

    skill_spec: SkillSpec

    @abstractmethod
    def forward(self, ctx: SkillContext) -> SkillContext:
        """Read from and write to a SkillContext."""
        raise NotImplementedError

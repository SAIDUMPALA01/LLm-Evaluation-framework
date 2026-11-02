"""sixdim-eval: evaluate LLM outputs on six dimensions and measure how far to trust the judge."""

from .agreement import judge_agreement, quadratic_weighted_kappa
from .consistency import output_consistency, score_stability, text_similarity
from .dimensions import ALL_DIMENSIONS, Dimension
from .judges import HeuristicJudge, LLMJudge
from .models import Case, Evaluation, Flag

__version__ = "0.1.0"

__all__ = [
    "ALL_DIMENSIONS",
    "Case",
    "Dimension",
    "Evaluation",
    "Flag",
    "HeuristicJudge",
    "LLMJudge",
    "judge_agreement",
    "output_consistency",
    "quadratic_weighted_kappa",
    "score_stability",
    "text_similarity",
]

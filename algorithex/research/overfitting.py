"""
How much of a backtest result is luck, measured rather than guessed.

A Sharpe ratio tells you what happened. It says nothing about how many other
parameter sets were tried before this one looked good, and that number is the
entire difference between an edge and a coincidence. Search 200 parameter
combinations, keep the best, and the winner's Sharpe is inflated by roughly
``sqrt(2 * ln(200))`` -- about 3.5 -- purely from the selection.

This module makes that cost explicit instead of implicit.

:func:`probability_of_backtest_overfitting`
    The combinatorially symmetric cross-validation procedure. Split the data
    into S blocks, form every way of choosing S/2 as "in-sample" and S/2 as
    "out-of-sample", and for each split ask whether the best in-sample
    configuration also ranked best out-of-sample. If it usually does not, the
    ranking is noise. The reported statistic is the fraction of splits where
    in-sample rank did *not* translate, which is a probability, not a score.

:func:`deflated_sharpe`
    The Sharpe ratio you should have believed, given that you tried N
    configurations and the best one is no longer special after accounting for
    the variance of the estimates themselves. It penalises both many trials and
    a noisy return series, which is what actually kills backtests.

:func:`plateau_analysis`
    The practical companion. A strategy whose profit sits on a narrow spike in
    parameter space is fitted; one whose profit sits on a broad plateau is
    probably real, because a broad plateau is what a genuine effect looks like
    once noise is smoothed over.

None of these need the trading engine. They take per-bar returns and the
per-configuration Sharpe ratios you already have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    'PboResult',
    'DeflatedSharpeResult',
    'PlateauResult',
    'sharpe_ratio',
    'probability_of_backtest_overfitting',
    'deflated_sharpe',
    'expected_max_sharpe',
    'plateau_analysis',
    'overfitting_report',
]

# Bailey, Borwein, Lopez de Prado and Zhu (2014), "The Deflated Sharpe Ratio".
_EULER_MASCHERONI = 0.5772156649015329


def _as_returns(returns) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f'returns must be one dimensional, got shape {values.shape}')
    return values


def sharpe_ratio(returns, periods_per_year: int = 365) -> float:
    """
    Annualised Sharpe ratio, using the population standard deviation.

    `ddof=0` rather than the usual `ddof=1` because the selection effect
    corrections downstream assume the same estimator consistently; mixing
    conventions is a quiet source of disagreement between two Sharpe ratios
    computed from identical data.
    """
    values = _as_returns(returns)
    if values.size < 2:
        return float('nan')
    deviation = float(values.std())
    # A constant series has no floating-point-exactly-zero standard deviation,
    # so the guard is relative to the size of the values. Without it a flat
    # return series reports a Sharpe of 7e16 instead of being infinite.
    if deviation <= 1e-12 * max(float(np.abs(values).max()), 1e-12):
        return float('inf') if values.mean() > 0 else float('-inf')
    return float(values.mean() / deviation * np.sqrt(periods_per_year))


@dataclass
class PboResult:
    """
    The CSCV verdict.

    logit
        The mean logit of the per-split out-of-sample relative rank. The
        logarithm is the right transform because the relative rank piles up
        near zero and one; the logit makes the middle of the range readable
        instead of compressed into a cliff. Negative means it transferred.
    """

    pbo: float
    logit: float
    n_splits: int
    n_configurations: int
    is_sqrt: float
    median_is_sharpe: float
    performance_degradation: float
    verdicts: List[float] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Plain-language read of the probability."""
        if self.pbo < 0.2:
            return 'UNLIKELY_TO_BE_OVERFIT'
        if self.pbo < 0.5:
            return 'AMBIGUOUS'
        if self.pbo < 0.8:
            return 'LIKELY_OVERFIT'
        return 'HIGHLY_LIKELY_OVERFIT'

    @property
    def concerning(self) -> bool:
        """Whether this warrants not deploying the strategy."""
        return self.pbo >= 0.5

    def as_dict(self) -> dict:
        return {
            'pbo': round(self.pbo, 6),
            'logit': round(self.logit, 6),
            'verdict': self.verdict,
            'n_splits': self.n_splits,
            'n_configurations': self.n_configurations,
            'performance_degradation': round(self.performance_degradation, 6),
        }


def _partition_matrix(returns_matrix: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Split each strategy's return stream into `n_blocks` contiguous blocks.

    Contiguous rather than shuffled on purpose: CSCV's argument is that a
    strategy which looks good in one stretch of time and bad in another has not
    found an edge, and shuffling destroys exactly that structure.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    n_strategies, n_bars = matrix.shape
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f'n_blocks must be an even integer >= 2, got {n_blocks}')
    if n_blocks > n_bars:
        raise ValueError(
            f'cannot split {n_bars} bars into {n_blocks} blocks'
        )
    block_size = n_bars // n_blocks
    trimmed = matrix[:, : block_size * n_blocks]
    return trimmed.reshape(n_strategies, n_blocks, block_size)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_blocks: int = 8,
    periods_per_year: int = 365,
) -> PboResult:
    """
    Probability that the best in-sample configuration is not the best
    out-of-sample.

    `returns_matrix` is one row per parameter configuration, so the shapes are
    the part people get wrong: rows are what you varied, columns are time.

    For every way of splitting the blocks in half, the configurations are
    ranked in-sample, and where the in-sample *best* lands in the
    out-of-sample ranking is recorded. A PBO of 0.5 means the in-sample ranking
    carried no information at all about the out-of-sample ranking, which is
    the signature of pure noise. A PBO near 0 means the ranking transfers; a
    PBO near 1 means the in-sample winner was reliably a loser afterwards.
    """
    blocks = _partition_matrix(returns_matrix, n_blocks)
    n_strategies, n_blocks_actual, block_size = blocks.shape
    if n_strategies < 2:
        raise ValueError(
            f'need at least 2 configurations to rank them, got {n_strategies}'
        )

    half = n_blocks_actual // 2
    all_splits = list(combinations(range(n_blocks_actual), half))
    # Both halves of each partition are used, so 2 * C(S, S/2) evaluations.
    partitions = all_splits + [
        tuple(i for i in range(n_blocks_actual) if i not in split) for split in all_splits
    ]

    logits: List[float] = []
    degradation: List[float] = []

    for split in partitions:
        mask = np.zeros(n_blocks_actual, dtype=bool)
        mask[list(split)] = True
        in_sample = blocks[:, mask, :].reshape(n_strategies, -1)
        out_sample = blocks[:, ~mask, :].reshape(n_strategies, -1)

        in_sharpe = np.array(
            [sharpe_ratio(row, periods_per_year) for row in in_sample]
        )
        out_sharpe = np.array(
            [sharpe_ratio(row, periods_per_year) for row in out_sample]
        )
        if not np.all(np.isfinite(in_sharpe)) or not np.all(np.isfinite(out_sharpe)):
            continue

        in_rank = stats.rankdata(-in_sharpe)
        out_rank = stats.rankdata(-out_sharpe)
        # rankdata(-sharpe) makes rank 1 the *best*, so the configuration that
        # looked best in-sample is the minimum rank, not the maximum.
        best_in_sample = float(np.min(in_rank))
        matching = out_rank[in_rank == best_in_sample]
        relative = float(np.mean(matching)) / (n_strategies + 1.0)

        # logit of the relative rank, so 0 and 1 map to +/-inf instead of NaN.
        # `relative` is the OOS rank of the IS-best divided by N+1, where rank 1
        # is the best OOS result. So relative < 0.5 means it landed in the top
        # half out-of-sample -- it transferred. The probability that it *failed*
        # to transfer is therefore the fraction above 0.5, not below.
        clipped = float(np.clip(relative, 1e-9, 1 - 1e-9))
        logits.append(float(np.log(clipped / (1.0 - clipped))))
        degradation.append(float(np.mean(out_sharpe) - np.mean(in_sharpe)))

    if not logits:
        raise ValueError(
            'no usable splits: every split produced a non-finite Sharpe ratio'
        )

    logits_array = np.array(logits)
    pbo = float(np.mean(logits_array > 0))
    return PboResult(
        pbo=pbo,
        logit=float(np.mean(logits_array)),
        n_splits=len(logits),
        n_configurations=int(n_strategies),
        is_sqrt=float(np.sqrt(np.mean(logits_array**2))),
        median_is_sharpe=float(np.median(logits_array)),
        performance_degradation=float(np.mean(degradation)),
        verdicts=logits,
    )


def expected_max_sharpe(n_trials: int, variance_ratio: float = 1.0) -> float:
    """
    Expected maximum Sharpe ratio from `n_trials` trials of pure noise.

    The Euler-Mascheroni correction is what makes the number real: the
    expected maximum of N standard normals grows like
    ``(1-gamma) * z_N + gamma * z_1``, and dropping the ``gamma * z_1`` term
    understates the selection effect by roughly half a Sharpe point.
    """
    if n_trials < 1:
        raise ValueError(f'n_trials must be at least 1, got {n_trials}')
    if variance_ratio <= 0:
        raise ValueError(f'variance_ratio must be positive, got {variance_ratio}')
    if n_trials == 1:
        # The maximum of a single draw is that draw, so there is no selection
        # effect to correct for.
        return 0.0
    z_n = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z_1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials + 0.5 / n_trials))
    return float(((1.0 - _EULER_MASCHERONI) * z_n + _EULER_MASCHERONI * z_1) * np.sqrt(variance_ratio))


@dataclass
class DeflatedSharpeResult:
    """The Sharpe you should have believed, next to the one you did."""

    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    n_trials: int
    probability_sharpe_is_positive: float
    trials_to_reach_observed: float

    @property
    def significant(self) -> bool:
        """
        Whether the edge survives deflation.

        The 0.95 threshold is the conventional one: below it, the observed
        Sharpe is not distinguishable from the best of the trials you ran.
        """
        return self.probability_sharpe_is_positive >= 0.95

    @property
    def verdict(self) -> str:
        if self.deflated_sharpe <= 0:
            return 'NO_EDGE_AFTER_DEFLATION'
        if not self.significant:
            return 'NOT_SIGNIFICANT'
        return 'SURVIVES_DEFLATION'

    def as_dict(self) -> dict:
        return {
            'observed_sharpe': round(self.observed_sharpe, 6),
            'deflated_sharpe': round(self.deflated_sharpe, 6),
            'expected_max_sharpe': round(self.expected_max_sharpe, 4),
            'probability_sharpe_is_positive': round(
                self.probability_sharpe_is_positive, 6
            ),
            'n_trials': self.n_trials,
            'verdict': self.verdict,
        }


def deflated_sharpe(
    returns,
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]] = None,
    periods_per_year: int = 365,
) -> DeflatedSharpeResult:
    """
    Deflated Sharpe ratio for one configuration.

    `n_trials` is how many configurations were tried in total -- including the
    ones that lost, which is the whole point. Passing the *number of trials
    that succeeded* instead is the single most common way to use this
    function wrongly, and it always flatters the result.

    `trial_sharpes`, when supplied, measures the cross-sectional variance of
    the trial Sharpe ratios directly, which is more accurate than assuming
    they are all drawn from the winner's own return series.
    """
    values = _as_returns(returns)
    if values.size < 2:
        raise ValueError(f'need at least 2 returns, got {values.size}')
    if n_trials < 1:
        raise ValueError(f'n_trials must be at least 1, got {n_trials}')

    observed = sharpe_ratio(values, periods_per_year)
    if not np.isfinite(observed):
        raise ValueError('observed Sharpe is not finite; returns may be constant')

    per_bar = observed / np.sqrt(periods_per_year)
    standard_error = float(np.std(values, ddof=1)) / np.sqrt(values.size) * np.sqrt(
        periods_per_year
    )
    if standard_error <= 0:
        raise ValueError('cannot estimate the Sharpe standard error')

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sharpes = np.asarray(trial_sharpes, dtype=np.float64)
        sharpes = sharpes[np.isfinite(sharpes)]
        variance_ratio = float(sharpes.var()) if sharpes.size > 1 else 1.0
        trials = max(int(sharpes.size), 1)
    else:
        # The winner's own variance is the conservative stand-in.
        variance_ratio = 1.0
        trials = int(n_trials)

    benchmark = expected_max_sharpe(trials, variance_ratio)
    deflated = observed - benchmark
    probability = float(stats.norm.cdf(deflated / standard_error))

    # How many noise trials would have been needed to expect this Sharpe.
    target = max(abs(observed) * np.sqrt(variance_ratio), 1e-9)
    exponent = (target / (1.0 - _EULER_MASCHERONI)) ** 2 / 2.0
    # Beyond ~700 the exponential overflows; the honest answer there is "more
    # trials than could ever have been run", which is itself the finding.
    trials_needed = float(np.exp(exponent)) if exponent < 700.0 else float('inf')

    return DeflatedSharpeResult(
        observed_sharpe=observed,
        deflated_sharpe=float(deflated),
        expected_max_sharpe=benchmark,
        n_trials=trials,
        probability_sharpe_is_positive=probability,
        trials_to_reach_observed=trials_needed,
    )


@dataclass
class PlateauResult:
    """Whether the parameter landscape is a spike or a plateau."""

    is_plateau: bool
    peak_value: float
    peak_ratio: float
    within_10pct: int
    total: int
    robustness_score: float
    grid: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if not self.is_plateau:
            return 'SPIKE_FITTED'
        return 'BROAD_PLATEAU'

    def as_dict(self) -> dict:
        return {
            'is_plateau': self.is_plateau,
            'verdict': self.verdict,
            'peak_value': round(self.peak_value, 6),
            'peak_ratio': round(self.peak_ratio, 6),
            'within_10pct': self.within_10pct,
            'total': self.total,
            'robustness_score': round(self.robustness_score, 6),
        }


def plateau_analysis(
    values,
    axes: Optional[Dict[str, Sequence[float]]] = None,
    min_relative_width: float = 0.10,
    min_fraction: float = 0.15,
) -> PlateauResult:
    """
    Is the best result on a broad plateau or a narrow spike?

    A grid of performance values (Sharpe ratios, typically) is examined: the
    peak is located, then the fraction of the grid within `min_relative_width`
    of it is measured. A genuine effect keeps producing decent results when its
    parameters are nudged, because the underlying phenomenon is still there;
    a fitted one only works at exactly the values that were searched.

    That fraction is the number to look at, and it is a direct, concrete answer
    to "did this survive being perturbed" -- the same question
    :mod:`algorithex.audit.robustness` asks of the data instead of the
    parameters.
    """
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f'values must be a non-empty 1-D grid, got shape {grid.shape}')
    if not np.all(np.isfinite(grid)):
        raise ValueError('grid contains non-finite values')
    if min_relative_width < 0:
        raise ValueError(
            f'min_relative_width must be non-negative, got {min_relative_width}'
        )
    if not 0 < min_fraction <= 1:
        raise ValueError(f'min_fraction must be in (0, 1], got {min_fraction}')

    peak = float(np.max(grid))
    absolute = min_relative_width * abs(peak)
    within = grid >= peak - absolute
    fraction = float(within.sum()) / grid.size

    return PlateauResult(
        is_plateau=bool(fraction >= min_fraction),
        peak_value=peak,
        peak_ratio=float(np.count_nonzero(within) / max(grid.size, 1)),
        within_10pct=int(within.sum()),
        total=int(grid.size),
        robustness_score=fraction,
        grid=dict(axes) if axes else {},
    )


def overfitting_report(
    returns_matrix: np.ndarray,
    best_returns,
    n_blocks: int = 8,
    n_trials: Optional[int] = None,
    periods_per_year: int = 365,
) -> Dict[str, object]:
    """
    The three overfitting checks in one call.

    Takes the full matrix of every configuration tried plus the returns of the
    one that was kept, and answers: how much of the ranking was noise, how much
    of the winner's Sharpe was selection, and did the result survive being
    moved. All three are read off the same data you already have.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    if matrix.shape[0] < 2:
        raise ValueError('need at least 2 configurations to assess overfitting')

    pbo = probability_of_backtest_overfitting(matrix, n_blocks, periods_per_year)

    trials = int(n_trials) if n_trials is not None else matrix.shape[0]
    trial_sharpes = [
        sharpe_ratio(row, periods_per_year) for row in matrix
    ]
    finite = [s for s in trial_sharpes if np.isfinite(s)]
    deflated = deflated_sharpe(
        best_returns,
        n_trials=trials,
        trial_sharpes=finite if len(finite) > 1 else None,
        periods_per_year=periods_per_year,
    )

    observed_grid = np.where(np.isfinite(trial_sharpes), trial_sharpes, np.nanmin(
        np.asarray(trial_sharpes)[np.isfinite(trial_sharpes)]
    ))
    plateau = plateau_analysis(observed_grid)

    return {
        'pbo': pbo.as_dict(),
        'deflated_sharpe': deflated.as_dict(),
        'plateau': plateau.as_dict(),
        'verdict': _overall_verdict(pbo, deflated, plateau),
    }


def _overall_verdict(pbo: PboResult, deflated: DeflatedSharpeResult, plateau: PlateauResult) -> str:
    """
    One word combining all three checks.

    Any single failure is enough to disqualify, because they fail
    independently: a strategy can have a healthy PBO, a deflated Sharpe above
    zero, and still be a spike that only works at one exact parameter value.
    """
    failures = []
    if pbo.concerning:
        failures.append('ranking does not transfer out-of-sample')
    if not deflated.significant:
        failures.append('Sharpe does not survive selection correction')
    if not plateau.is_plateau:
        failures.append('result sits on a spike in parameter space')

    if not failures:
        return 'PASSES_OVERFITTING_CHECKS'
    return 'FAILS: ' + '; '.join(failures)

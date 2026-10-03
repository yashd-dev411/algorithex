"""
One number, one verdict, and the reasoning behind both.

A strategy passes through five kinds of question before anyone should risk
money on it, and the tools for each already exist elsewhere in this codebase:

Does it cheat?
    :mod:`algorithex.audit.leakage` finds strategies that read the future.
Is the result luck?
    :mod:`algorithex.research.overfitting` measures how much of the Sharpe is
    selection.
Does it survive being wrong?
    :mod:`algorithex.audit.robustness` attacks the data and searches for the
    worst price path.
Does it survive its own costs?
    :mod:`algorithex.execution.costs` charges what execution really costs.
Does it hold up out of sample?
    :mod:`algorithex.research.walkforward` scores it on data it never saw.

Running those is easy; the hard part is that five separate reports are read by
nobody. This module is the thing you can put in front of a decision: it runs
whichever checks you give it inputs for, grades the result, and names the
specific findings that would stop deployment.

Two properties are deliberate and worth stating up front.

Missing evidence is not a pass
    A check you supplied no input for is reported as ``not_run`` and counts
    against the grade. A certification built from three of five checks says so
    rather than quietly scoring as though five were done.

One blocking failure is enough
    Look-ahead is not offset by a good Sharpe. Blocking findings cap the grade
    outright, so a strategy cannot average its way past having cheated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    'Check',
    'CertificationReport',
    'GRADES',
    'certify',
    'grade_for',
    'render_report',
]

PASS = 'pass'
WARN = 'warn'
FAIL = 'fail'
NOT_RUN = 'not_run'

GRADES = ('A', 'B', 'C', 'D', 'F')


@dataclass(frozen=True)
class Check:
    """
    One verdict on one dimension.

    `weight` is how much the check counts towards the score. `blocking` marks
    the checks that cannot be averaged away -- a look-ahead bug or a strategy
    that loses money are not compensable.
    """

    name: str
    status: str
    score: float
    detail: str
    weight: float = 1.0
    blocking: bool = False
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in (PASS, WARN, FAIL, NOT_RUN):
            raise ValueError(
                f'status must be one of {PASS!r}, {WARN!r}, {FAIL!r}, {NOT_RUN!r}, '
                f'got {self.status!r}'
            )
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f'score must be in [0, 1], got {self.score}')
        if self.weight <= 0:
            raise ValueError(f'weight must be positive, got {self.weight}')

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> dict:
        return {
            'name': self.name,
            'status': self.status,
            'score': round(self.score, 4),
            'weight': self.weight,
            'blocking': self.blocking,
            'detail': self.detail,
        }


def _clamp(value: float) -> float:
    """Map any real number into [0, 1], so callers can be careless safely."""
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


@dataclass
class CertificationReport:
    """The full picture: a grade, a verdict, and every check behind it."""

    strategy_name: str
    checks: List[Check]

    @property
    def score(self) -> float:
        """
        Weighted mean of the checks that actually ran.

        ``not_run`` checks are included with a score of zero rather than
        dropped. Leaving them out would let you improve your grade by
        running fewer tests, which is exactly backwards.
        """
        if not self.checks:
            return 0.0
        total = sum(c.weight for c in self.checks)
        return float(sum(c.score * c.weight for c in self.checks) / total)

    @property
    def blocking_failures(self) -> List[Check]:
        return [c for c in self.checks if c.blocking and c.status == FAIL]

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def not_run(self) -> List[Check]:
        return [c for c in self.checks if c.status == NOT_RUN]

    @property
    def coverage(self) -> float:
        """Fraction of the weighted checks that actually produced a verdict."""
        if not self.checks:
            return 0.0
        return float(sum(c.weight for c in self.checks if c.status != NOT_RUN) / sum(c.weight for c in self.checks))

    @property
    def grade(self) -> str:
        return grade_for(self.score, bool(self.blocking_failures))

    @property
    def verdict(self) -> str:
        """
        What to actually do about it.

        ``REJECTED`` for any blocking failure or an F; ``CONDITIONAL`` for
        anything below a B, so a C can never read as a go-ahead; only a clean
        A or B certifies.
        """
        if self.blocking_failures:
            return 'REJECTED'
        grade = self.grade
        if grade == 'F':
            return 'REJECTED'
        if grade in ('C', 'D'):
            return 'CONDITIONAL'
        return 'CERTIFIED'

    @property
    def deployable(self) -> bool:
        return self.verdict == 'CERTIFIED'

    def reasons(self) -> List[str]:
        """Plain sentences a person can act on, worst first."""
        lines = []
        for check in self.blocking_failures:
            lines.append(f'BLOCKING - {check.name}: {check.detail}')
        for check in self.failures:
            lines.append(f'FAILED - {check.name}: {check.detail}')
        for check in self.warnings:
            lines.append(f'WARNING - {check.name}: {check.detail}')
        for check in self.not_run:
            lines.append(f'NOT RUN - {check.name}: {check.detail}')
        return lines

    def check(self, name: str) -> Check:
        """Look a check up by name, with a useful error if it is missing."""
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(
            f'no check named {name!r}; available: '
            f'{[c.name for c in self.checks]}'
        )

    def as_dict(self) -> dict:
        return {
            'strategy': self.strategy_name,
            'grade': self.grade,
            'verdict': self.verdict,
            'score': round(self.score, 4),
            'coverage': round(self.coverage, 4),
            'checks': [c.as_dict() for c in self.checks],
            'reasons': self.reasons(),
        }

    def render(self) -> str:
        return render_report(self)


def grade_for(score: float, has_blocking_failure: bool = False) -> str:
    """
    Map a 0-1 score onto a letter grade.

    `has_blocking_failure` caps the grade at F regardless of score, so a
    look-ahead bug cannot be offset by an excellent Sharpe.
    """
    value = _clamp(score)
    if has_blocking_failure:
        return 'F'
    if value >= 0.85:
        return 'A'
    if value >= 0.70:
        return 'B'
    if value >= 0.55:
        return 'C'
    if value >= 0.40:
        return 'D'
    return 'F'


def _check_leakage(source_path: Optional[str], source_code: Optional[str]) -> Check:
    """
    Static look-ahead detection.

    Blocking and unforgiving: a strategy that reads the future is not a
    strategy with a flaw, it is a measurement of the future, and no amount of
    performance data makes that tradeable.
    """
    if not source_path and not source_code:
        return Check(
            name='lookahead',
            status=NOT_RUN,
            score=0.0,
            detail='no strategy source supplied; run detect_lookahead() on it',
            weight=1.5,
            blocking=True,
        )

    from ..audit.leakage import audit_source_file, detect_lookahead

    try:
        if source_code:
            audit = detect_lookahead(source_code)
        else:
            audit = audit_source_file(source_path)
    except Exception as exc:  # noqa: BLE001 - a broken check must not crash the audit
        return Check(
            name='lookahead',
            status=FAIL,
            score=0.0,
            detail=f'could not analyse source: {exc}',
            weight=1.5,
            blocking=True,
        )

    findings = list(audit.errors)
    if not findings:
        soft = list(audit.warnings)
        if soft:
            return Check(
                name='lookahead',
                status=WARN,
                score=0.6,
                detail=f'{len(soft)} suspicious pattern(s) worth reviewing: '
                + ', '.join(str(f) for f in soft[:2]),
                weight=1.5,
                blocking=True,
            )
        return Check(
            name='lookahead',
            status=PASS,
            score=1.0,
            detail='no look-ahead patterns found',
            weight=1.5,
            blocking=True,
        )
    return Check(
        name='lookahead',
        status=FAIL,
        score=0.0,
        detail=f'{len(findings)} look-ahead pattern(s): '
        + ', '.join(str(f) for f in findings[:3]),
        weight=1.5,
        blocking=True,
        evidence={'findings': [str(f) for f in findings]},
    )


def _check_overfitting(matrix: Optional[np.ndarray], best_returns) -> Check:
    """
    PBO, deflated Sharpe and plateau, combined into one score.

    Scored on the probability rather than the verdict string, so a PBO of 0.3
    and a PBO of 0.49 are visibly different rather than both being "ambiguous".
    """
    if matrix is None or best_returns is None:
        return Check(
            name='overfitting',
            status=NOT_RUN,
            score=0.0,
            detail='supply the returns of every configuration tried, plus the best one',
            weight=1.5,
        )

    from .overfitting import overfitting_report

    try:
        report = overfitting_report(matrix, best_returns)
    except ValueError as exc:
        return Check(
            name='overfitting',
            status=NOT_RUN,
            score=0.0,
            detail=f'could not be assessed: {exc}',
            weight=1.5,
        )

    pbo = float(report['pbo']['pbo'])
    deflated = float(report['deflated_sharpe']['deflated_sharpe'])
    plateau = bool(report['plateau']['is_plateau'])

    # Three roughly equal thirds: how well the ranking transfers, whether
    # anything survives selection, and whether the result is broad or a spike.
    ranking_score = _clamp(1.0 - pbo)
    selection_score = _clamp(deflated / 3.0)
    plateau_score = float(report['plateau']['robustness_score'])
    score = 0.4 * ranking_score + 0.35 * selection_score + 0.25 * plateau_score

    status = PASS if score >= 0.6 else (WARN if score >= 0.35 else FAIL)
    return Check(
        name='overfitting',
        status=status,
        score=score,
        detail=(
            f'PBO {pbo:.2f}, deflated Sharpe {deflated:.2f}, '
            f'{"broad plateau" if plateau else "spike-fitted"}'
        ),
        weight=1.5,
        evidence=report,
    )


def _missing_inputs_detail(prices, strategy) -> str:
    """
    Say precisely what is missing.

    "supply a strategy and a price series" is unhelpful when the caller
    already supplied one of them; naming the specific absent input is the
    difference between a finding you can act on and a shrug.
    """
    if strategy is None and prices is None:
        return 'supply a strategy(prices) -> pnl callable and a price series'
    if strategy is None:
        return (
            'prices supplied but no executable strategy: pass '
            'strategy=strategy_fn so the attacks have something to attack'
        )
    return 'supply a price series to attack'


def _check_robustness(strategy, prices) -> Check:
    """
    Adversarial falsification, scored on how much profit survives.

    Blocking only when the strategy loses money at all, which is a different
    failure from being fragile.
    """
    if strategy is None or prices is None:
        return Check(
            name='robustness',
            status=NOT_RUN,
            score=0.0,
            detail=_missing_inputs_detail(prices, strategy),
            weight=1.0,
        )

    from ..audit.robustness import audit_robustness

    try:
        report = audit_robustness(strategy, prices)
    except (ValueError, TypeError) as exc:
        return Check(
            name='robustness', status=NOT_RUN, score=0.0,
            detail=f'could not be assessed: {exc}', weight=1.0,
        )

    if report.baseline_pnl <= 0:
        return Check(
            name='robustness',
            status=FAIL,
            score=0.0,
            detail=f'strategy loses money on the historical path ({report.baseline_pnl:.4f})',
            weight=1.0,
            blocking=True,
            evidence=report.as_dict(),
        )

    score = 0.5 * _clamp(report.survival_rate) + 0.5 * _clamp(report.retention / 2.0)
    status = PASS if score >= 0.7 else (WARN if score >= 0.45 else FAIL)
    return Check(
        name='robustness',
        status=status,
        score=score,
        detail=(
            f'{report.survival_rate:.0%} of attacks survived, '
            f'{report.retention:.0%} of PnL retained, '
            f'break-even cost {report.break_even_cost:.1f}x'
        ),
        weight=1.0,
        evidence=report.as_dict(),
    )


def _check_costs(strategy, prices, required_multiple: float) -> Check:
    """
    Does the edge survive a realistic multiple of the assumed costs?

    `required_multiple` is how many times the assumed fees and slippage would
    have to be before the strategy stops working. Three is the usual bar: it
    means the result does not depend on the cost model being exactly right.
    """
    if strategy is None or prices is None:
        return Check(
            name='cost_survival',
            status=NOT_RUN,
            score=0.0,
            detail=_missing_inputs_detail(prices, strategy),
            weight=1.0,
        )

    from ..audit.robustness import break_even_multiple

    try:
        multiple = float(break_even_multiple(strategy, prices))
    except (ValueError, TypeError) as exc:
        return Check(
            name='cost_survival', status=NOT_RUN, score=0.0,
            detail=f'could not be assessed: {exc}', weight=1.0,
        )

    if not np.isfinite(multiple):
        status, score, detail = PASS, 1.0, 'no cost can erase the edge'
    elif multiple >= required_multiple:
        score = _clamp(multiple / (required_multiple * 2))
        status, detail = PASS, f'break-even at {multiple:.1f}x assumed costs'
    elif multiple >= 1.0:
        score = _clamp(multiple / required_multiple)
        status = WARN if score >= 0.4 else FAIL
        detail = f'break-even at only {multiple:.1f}x assumed costs'
    else:
        status, score = FAIL, 0.0
        detail = 'unprofitable before any cost is charged'

    return Check(
        name='cost_survival',
        status=status,
        score=score,
        detail=detail,
        weight=1.0,
        evidence={'break_even_multiple': multiple, 'required': required_multiple},
    )


def _check_out_of_sample(window_results: Optional[Sequence[dict]]) -> Check:
    """
    Walk-forward consistency, scored on the *worst* window rather than the mean.

    The mean of a walk-forward distribution is the one statistic nobody should
    quote, because a single good window drags it up and hides the fact that the
    strategy only works in one regime.
    """
    if not window_results:
        return Check(
            name='out_of_sample',
            status=NOT_RUN,
            score=0.0,
            detail='supply evaluate_windows() results from walk_forward_evaluate',
            weight=1.5,
        )

    sharpes = [
        float(w.get('sharpe', float('nan')))
        for w in window_results
        if isinstance(w, dict)
    ]
    sharpes = [s for s in sharpes if np.isfinite(s)]
    if not sharpes:
        return Check(
            name='out_of_sample', status=NOT_RUN, score=0.0,
            detail='no finite Sharpe ratios in the supplied windows', weight=1.5,
        )

    positive_fraction = float(np.mean([s > 0 for s in sharpes]))
    worst = float(min(sharpes))
    median = float(np.median(sharpes))

    score = 0.6 * positive_fraction + 0.4 * _clamp(median / max(abs(median), 1.0) if median > 0 else 0.0)
    if median <= 0:
        score = 0.2 * positive_fraction
    elif worst < 0:
        # A strategy that is deeply negative in its worst window is fragile
        # even if the median looks fine.
        score = min(score, 0.65)

    status = PASS if score >= 0.7 else (WARN if score >= 0.45 else FAIL)
    return Check(
        name='out_of_sample',
        status=status,
        score=score,
        detail=(
            f'{positive_fraction:.0%} of {len(sharpes)} windows profitable, '
            f'median Sharpe {median:.2f}, worst {worst:.2f}'
        ),
        weight=1.5,
        evidence={'windows': len(sharpes), 'worst_sharpe': worst, 'median_sharpe': median},
    )


def certify(
    strategy_name: str = 'unnamed strategy',
    source_path: Optional[str] = None,
    source_code: Optional[str] = None,
    strategy=None,
    prices=None,
    configuration_returns: Optional[np.ndarray] = None,
    best_returns=None,
    window_results: Optional[Sequence[dict]] = None,
    required_cost_multiple: float = 3.0,
) -> CertificationReport:
    """
    Run every check the inputs support and grade the result.

    Pass only what you have; everything else is reported as ``not_run`` and
    counted against the grade. Passing everything produces the strictest
    report, which is the correct incentive.

    `configuration_returns` is the matrix of every parameter configuration you
    tried, one row each. Omitting it is the most tempting shortcut here and it
    disables the check that most reliably catches an overfit strategy, because
    without knowing what else was tried there is no way to know whether the
    winner is special.
    """
    return CertificationReport(
        strategy_name=strategy_name,
        checks=[
            _check_leakage(source_path, source_code),
            _check_overfitting(configuration_returns, best_returns),
            _check_robustness(strategy, prices),
            _check_costs(strategy, prices, required_cost_multiple),
            _check_out_of_sample(window_results),
        ],
    )


def render_report(report: CertificationReport) -> str:
    """
    A fixed-width text report.

    Deliberately plain: this is meant to be pasted into a pull request or an
    issue, where a table is more useful than a JSON blob.
    """
    symbols = {PASS: 'PASS', WARN: 'WARN', FAIL: 'FAIL', NOT_RUN: '----'}
    lines = [
        '=' * 72,
        f'Strategy certification: {report.strategy_name}',
        '=' * 72,
        f'Grade: {report.grade}    Verdict: {report.verdict}'
        f'    Score: {report.score:.2f}    Coverage: {report.coverage:.0%}',
        '-' * 72,
    ]
    for check in report.checks:
        mark = '!!' if check.blocking and check.status == FAIL else '  '
        lines.append(
            f'{mark} [{symbols[check.status]}] {check.name:<16} '
            f'{check.score:>5.2f}  {check.detail}'
        )
    lines.append('-' * 72)
    reasons = report.reasons()
    if reasons:
        lines.append('Findings:')
        lines.extend(f'  {line}' for line in reasons)
    else:
        lines.append('No findings.')
    lines.append('=' * 72)
    return '\n'.join(lines)

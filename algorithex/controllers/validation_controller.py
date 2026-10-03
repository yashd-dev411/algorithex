"""
HTTP surface for the strategy validation tooling.

The audit, robustness and overfitting modules are pure computation, which is
what makes them easy to test and easy to forget. This controller is the part
that makes them reachable: a strategy file goes in, a verdict comes out, over
the same authenticated API as everything else in the dashboard.

Deliberately no strategy execution. Handing this service a strategy object
would mean importing arbitrary user code into the web process, so the callers
stay on the Python side and this layer only runs analysis over data and
source text.
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from algorithex.services.auth import require_auth

router = APIRouter(prefix="/validate", tags=["Validation"], dependencies=[Depends(require_auth)])

# Long enough to stop a trivial request being used to spin the CPU.
MAX_POINTS = 500_000
MAX_TEXT_CHARS = 2_000_000


class SourceRequestJson(BaseModel):
    """Scan strategy source text for look-ahead."""

    source: str
    source_name: Optional[str] = None


class OverfittingRequestJson(BaseModel):
    """
    Assess how much of a result is selection rather than signal.

    `configurations` is the matrix of every parameter combination that was
    tried, one row each. `best` is the returns of the one that was kept.
    Supplying only `best` is the trap this endpoint exists to make obvious:
    without knowing what else lost, there is no way to tell whether the winner
    is special.
    """

    configurations: List[List[float]]
    best: List[float]
    n_blocks: int = 8


def _as_series(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise HTTPException(400, f'{label} must be a flat list of numbers')
    if array.size < 3:
        raise HTTPException(400, f'{label} needs at least 3 values, got {array.size}')
    if array.size > MAX_POINTS:
        raise HTTPException(400, f'{label} exceeds {MAX_POINTS} points')
    if not np.all(np.isfinite(array)):
        raise HTTPException(400, f'{label} contains non-finite values')
    return array


@router.post('/lookahead')
def lookahead(json_request: SourceRequestJson) -> JSONResponse:
    """
    Statically scan strategy source for constructs that read the future.

    A strategy that looks ahead is not a strategy with a flaw: it is a
    measurement of the future, and no amount of performance data makes it
    tradeable. So this is reported as pass or fail, never as a score to be
    averaged against a good Sharpe.
    """
    from ..audit.leakage import detect_lookahead

    source = json_request.source or ''
    if not source.strip():
        raise HTTPException(400, 'source is empty')
    if len(source) > MAX_TEXT_CHARS:
        raise HTTPException(400, f'source exceeds {MAX_TEXT_CHARS} characters')

    audit = detect_lookahead(source, json_request.source_name or '<uploaded>')
    return JSONResponse(
        {
            'clean': audit.clean,
            'errors': [str(f) for f in audit.errors],
            'warnings': [str(f) for f in audit.warnings],
            'lines_examined': audit.lines_examined,
            'source_name': audit.source_name,
            'verdict': 'PASS' if audit.clean else 'FAIL',
        },
        status_code=200,
    )


@router.post('/overfitting')
def overfitting(json_request: OverfittingRequestJson) -> JSONResponse:
    """
    Probability of backtest overfitting, deflated Sharpe and plateau analysis.

    Read the numbers together. A PBO near 0 with a deflated Sharpe above zero
    and a broad plateau is the only combination worth deploying on.
    """
    from ..research.overfitting import overfitting_report

    matrix = np.asarray(json_request.configurations, dtype=np.float64)
    if matrix.ndim != 2:
        raise HTTPException(400, 'configurations must be a list of equal-length rows')
    if matrix.shape[0] < 2:
        raise HTTPException(400, 'need at least 2 configurations to rank them')
    if matrix.shape[1] > MAX_POINTS:
        raise HTTPException(400, f'configurations exceeds {MAX_POINTS} bars')
    if not np.all(np.isfinite(matrix)):
        raise HTTPException(400, 'configurations contain non-finite values')

    best = _as_series(json_request.best, 'best')
    if best.size != matrix.shape[1]:
        raise HTTPException(
            400,
            f'best has {best.size} bars but configurations have {matrix.shape[1]}',
        )

    n_blocks = json_request.n_blocks
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise HTTPException(400, f'n_blocks must be an even integer >= 2, got {n_blocks}')
    if n_blocks > matrix.shape[1]:
        raise HTTPException(
            400, f'cannot split {matrix.shape[1]} bars into {n_blocks} blocks'
        )

    try:
        report = overfitting_report(matrix, best, n_blocks=n_blocks)
    except ValueError as exc:
        # A degenerate grid is the caller's problem, not a server fault, and
        # saying so is more useful than a 500.
        raise HTTPException(400, str(exc))

    return JSONResponse(report, status_code=200)

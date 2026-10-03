"""
Correlation-aware capital allocation.

Equal weight is the default most people fall back to, and it is quietly wrong
in exactly the situation diversification matters: when assets correlate, the
notional 10% you gave the volatile one is not a 10% risk bet, it is most of the
portfolio's total risk. Risk parity fixes that by equalising *risk* rather than
*capital*.

Two things make the difference between this working and producing confident
nonsense:

Covariance is estimated from data you do not have
    A sample covariance matrix from 30 observations of 10 assets is singular,
    and the optimiser will happily exploit that instability to produce extreme
    weights. :func:`shrunk_covariance` blends the sample towards a constant
    correlation target, which is the cheapest available defence.

Risk parity is a fixed point, not a closed form
    There is no matrix formula for it. :func:`risk_parity_weights` iterates,
    and the result is only trustworthy if the resulting risk contributions are
    actually equal. :func:`risk_contributions` is what makes that checkable
    instead of something you have to take on faith.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@contextlib.contextmanager
def _quiet_slsqp():
    """
    Silence SLSQP's internal "values outside bounds" warnings.

    SLSQP evaluates trial points outside the box and clips them on the way back
    in; that is normal operation for the solver, not a problem with the caller,
    and it would otherwise print on every portfolio allocation. Scoped tightly
    to the solver call so genuine numerical warnings still surface.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Values in x were outside bounds during a minimize step',
            category=RuntimeWarning,
        )
        yield

__all__ = [
    'constant_correlation_covariance',
    'shrunk_covariance',
    'equal_weight_weights',
    'minimum_variance_weights',
    'risk_parity_weights',
    'portfolio_volatility',
    'risk_contributions',
    'diversification_ratio',
]


def _as_covariance(covariance) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'covariance must be square, got shape {matrix.shape}')
    if matrix.shape[0] == 0:
        raise ValueError('covariance must not be empty')
    return matrix


def constant_correlation_covariance(volatilities, correlation: float = 0.3) -> np.ndarray:
    """
    Build a covariance matrix with uniform off-diagonal correlation.

    This is the shrinkage target: a matrix with the right volatilities but no
    spurious correlation structure. Shrinking towards it cannot invent a
    relationship the data never supported.

    :param volatilities: per-asset standard deviations
    :param correlation: the single correlation assumed between every pair
    """
    vols = np.asarray(volatilities, dtype=np.float64).reshape(-1)
    n = vols.size
    if n == 0:
        raise ValueError('need at least one volatility')
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f'correlation must be within [-1, 1], got {correlation}')

    matrix = correlation * np.outer(vols, vols)
    if n > 1:
        # The diagonal is an asset's variance, not a correlation of an asset
        # with itself, so it must not pick up the `correlation` factor.
        np.fill_diagonal(matrix, 0.0)
    matrix[np.diag_indices_from(matrix)] = np.square(vols)
    return matrix


def shrunk_covariance(returns, shrinkage: float = 0.3, correlation: float = 0.3) -> np.ndarray:
    """
    Sample covariance blended towards a constant-correlation target.

    :param returns: (T, N) per-bar returns, one column per asset
    :param shrinkage: 0 keeps the raw sample matrix, 1 uses the target alone
    :param correlation: correlation assumed between every pair in the target
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f'returns must be 2D, got shape {matrix.shape}')
    if matrix.shape[0] < 2:
        raise ValueError(f'need at least 2 observations, got {matrix.shape[0]}')
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f'shrinkage must be within [0, 1], got {shrinkage}')

    sample = np.cov(matrix, rowvar=False, ddof=1)
    sample = np.atleast_2d(sample)
    target = constant_correlation_covariance(np.sqrt(np.diag(sample)), correlation)

    blended = (1.0 - shrinkage) * sample + shrinkage * target
    return _nearest_positive(blended)


def _nearest_positive(matrix: np.ndarray) -> np.ndarray:
    """
    Project a covariance matrix back onto the positive-definite cone.

    Eigenvalues are clipped to a small positive floor and the matrix rebuilt.
    That is the nearest positive-definite matrix in the Frobenius norm, which
    perturbs the estimate as little as possible.

    Simply nudging the diagonal upward until Cholesky succeeds also "works",
    but it distorts variances by an arbitrary amount and can fail outright on
    a matrix that is badly indefinite, so it is not used here.

    :param matrix: symmetric (N, N) matrix
    :return: the nearest positive-definite matrix
    """
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)

    floor = max(1e-12, float(np.abs(eigenvalues).max())) * 1e-10
    clipped = np.maximum(eigenvalues, floor)
    repaired = (eigenvectors * clipped) @ eigenvectors.T

    return 0.5 * (repaired + repaired.T)


def equal_weight_weights(n_assets: int) -> np.ndarray:
    """Capital weights of ``1/n`` each."""
    if n_assets < 1:
        raise ValueError(f'n_assets must be at least 1, got {n_assets}')
    return np.full(n_assets, 1.0 / n_assets)


def minimum_variance_weights(covariance, max_weight: Optional[float] = None) -> np.ndarray:
    """
    Long-only minimum-variance weights.

    Unconstrained, the answer is the closed form ``w ∝ Σ⁻¹ · 1``, which is
    used whenever it comes out long-only. When it does not (strongly
    correlated assets can make the unconstrained optimum short) or when a cap
    is set, the problem is handed to SLSQP, which always returns a feasible
    long-only point.

    :param covariance: (N, N) covariance matrix
    :param max_weight: optional per-asset cap, e.g. 0.25 to forbid concentration
    """
    cov = _as_covariance(covariance)
    n = cov.shape[0]

    cap = 1.0 if max_weight is None else float(max_weight)
    if cap < 0:
        raise ValueError(f'max_weight must be non-negative, got {max_weight}')
    if cap * n < 1.0 - 1e-12:
        raise ValueError(
            f'max_weight {cap} is infeasible for {n} assets; it must allow at '
            f'least {1.0 / n:.4f} each'
        )

    if max_weight is None:
        try:
            raw = np.linalg.solve(cov, np.ones(n))
        except np.linalg.LinAlgError:
            raw = None
        if raw is not None and np.all(raw > 0) and np.all(np.isfinite(raw)):
            return raw / raw.sum()

    def variance(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    with _quiet_slsqp():
        result = minimize(
            variance, np.full(n, 1.0 / n), method='SLSQP',
            bounds=[(0.0, cap)] * n,
            constraints=[{'type': 'eq', 'fun': lambda w: float(w.sum() - 1.0)}],
            options={'maxiter': 1000, 'ftol': 1e-14},
        )

    weights = np.clip(result.x, 0.0, cap)
    total = weights.sum()
    if total <= 0 or not np.all(np.isfinite(weights)):
        return equal_weight_weights(n)
    return weights / total


def risk_parity_weights(
    covariance,
    target_contributions: Optional[np.ndarray] = None,
    restarts: int = 6,
    random_state: int = 0,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Weights that equalise each asset's share of total portfolio risk.

    Solved by minimising the squared deviation of realised risk shares from the
    target budgets, which is zero exactly at the risk budgeting solution. The
    objective is solved with SLSQP from several starting points and the best
    result kept, because the surface has flat regions that a single start can
    stall in.

    Fixed-point iterations are the textbook alternative and were tried here
    first; they oscillate between equal weights and the solution for
    uncorrelated assets, so they were not kept.

    :param covariance: (N, N) covariance matrix
    :param target_contributions: per-asset share of total risk; defaults to
        equal risk (the defining property of risk parity)
    :param restarts: starting points to try; more is safer, each is cheap
    :param random_state: seed for the randomised starting points
    :param tol: convergence tolerance on the risk deviation
    """
    cov = _as_covariance(covariance)
    n = cov.shape[0]

    if target_contributions is None:
        budgets = np.full(n, 1.0 / n)
    else:
        budgets = np.asarray(target_contributions, dtype=np.float64).reshape(-1)
        if budgets.size != n:
            raise ValueError(
                f'target_contributions has {budgets.size} entries but there are {n} assets'
            )
        if np.any(budgets <= 0):
            raise ValueError('target_contributions must all be positive')
        budgets = budgets / budgets.sum()

    volatility = np.sqrt(np.diag(cov))
    if np.any(volatility <= 0):
        raise ValueError('covariance has a zero-variance asset, risk is undefined')

    def deviation(weights: np.ndarray) -> float:
        variance = float(weights @ cov @ weights)
        if not np.isfinite(variance) or variance <= 0:
            return np.inf
        shares = weights * (cov @ weights) / variance
        return float(np.sum(np.square(shares - budgets)))

    constraint = {'type': 'eq', 'fun': lambda w: float(w.sum() - 1.0)}
    bounds = [(1e-9, 1.0)] * n

    rng = np.random.default_rng(random_state)
    best_weights = np.full(n, 1.0 / n)
    best_score = np.inf

    for attempt in range(max(1, restarts)):
        if attempt == 0:
            start = np.full(n, 1.0 / n)
        elif attempt == 1:
            # Inverse-volatility start: a good guess for diagonal covariance.
            start = (1.0 / volatility) / np.sum(1.0 / volatility)
        else:
            start = rng.random(n) + 0.1
            start /= start.sum()

        try:
            with _quiet_slsqp():
                result = minimize(
                    deviation, start, method='SLSQP', bounds=bounds,
                    constraints=[constraint], options={'maxiter': 1000, 'ftol': tol},
                )
        except (ValueError, np.linalg.LinAlgError):
            continue

        candidate = np.clip(result.x, 0.0, 1.0)
        total = candidate.sum()
        if total <= 0 or not np.all(np.isfinite(candidate)):
            continue
        candidate /= total

        score = deviation(candidate)
        if score < best_score:
            best_score = score
            best_weights = candidate

    return best_weights


def portfolio_volatility(weights, covariance, periods_per_year: int = 1) -> float:
    """
    Volatility of a weighted portfolio.

    :param periods_per_year: annualise the result
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    cov = _as_covariance(covariance)
    if w.size != cov.shape[0]:
        raise ValueError(f'{w.size} weights do not match a {cov.shape[0]}-asset covariance')
    variance = float(w @ cov @ w)
    return float(np.sqrt(max(variance, 0.0)) * np.sqrt(periods_per_year))


def risk_contributions(weights, covariance) -> np.ndarray:
    """
    Each asset's share of total portfolio risk.

    Risk contribution is ``w_i * (Σw)_i / σ_p``: what the asset would hand
    back to portfolio volatility if it were removed. These sum to total
    volatility, so dividing by it gives contributions that sum to one — which
    is the form to read when checking whether risk parity actually worked.

    :return: absolute risk contributions, summing to portfolio volatility
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    cov = _as_covariance(covariance)
    if w.size != cov.shape[0]:
        raise ValueError(f'{w.size} weights do not match a {cov.shape[0]}-asset covariance')

    sigma_p = portfolio_volatility(w, cov)
    if sigma_p <= 0:
        return np.zeros_like(w)
    marginal = (cov @ w) / sigma_p
    return w * marginal


def diversification_ratio(weights, covariance) -> float:
    """
    Weighted average volatility divided by portfolio volatility.

    1.0 means no diversification benefit at all: the assets behave as one
    position. The higher the number, the less the portfolio moves than its
    parts imply.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    cov = _as_covariance(covariance)
    individual = w @ np.sqrt(np.diag(cov))
    portfolio = portfolio_volatility(w, cov)
    if portfolio <= 0:
        return 1.0
    return float(individual / portfolio)
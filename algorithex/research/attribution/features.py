"""
Feature-level attribution.

"Which of my twenty indicators is doing the work?" is the question that decides
whether a strategy is worth trading or just worth backtesting. A strategy whose
edge comes from one feature and nineteen noise generators is a strategy whose
edge a small parameter change destroys.

Two views are provided, because they fail differently:

Univariate
    Correlation of each feature with the outcome. Cheap, assumption-free, and
    completely unable to tell you whether two features are telling you the same
    thing twice.

Multivariate
    A ridge fit that partitions the outcome across correlated features. Ridge
    rather than ordinary least squares because indicator features are heavily
    collinear and the unregularised fit would hand the entire edge to whichever
    two features happen to be most collinear, which is an artefact rather than
    a finding.

Attribution is descriptive, not causal. It describes how a model distributed
credit across its inputs on the data it saw; it does not license the conclusion
that removing a feature with a small share would have changed nothing.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

__all__ = [
    'univariate_attribution',
    'ridge_attribution',
    'attribution_report',
]


def _as_matrix(features: Mapping[str, Sequence]) -> np.ndarray:
    if not features:
        raise ValueError('no features supplied')
    names = list(features)
    columns = [np.asarray(features[name], dtype=np.float64).reshape(-1) for name in names]
    lengths = {c.size for c in columns}
    if len(lengths) != 1:
        raise ValueError(f'features have differing lengths: {sorted(lengths)}')
    matrix = np.column_stack(columns)
    if matrix.shape[0] < 2:
        raise ValueError(f'need at least 2 observations, got {matrix.shape[0]}')
    return matrix


def _check_outcomes(outcomes: Sequence, n_rows: int) -> np.ndarray:
    y = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    if y.size != n_rows:
        raise ValueError(f'{n_rows} feature rows but {y.size} outcomes')
    if not np.all(np.isfinite(y)):
        raise ValueError('outcomes contain NaN or infinite values')
    return y


def _clean(matrix: np.ndarray, y: np.ndarray):
    """Drop observations any feature could not supply."""
    keep = np.all(np.isfinite(matrix), axis=1) & np.isfinite(y)
    return matrix[keep], y[keep], keep


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def univariate_attribution(
    features: Mapping[str, Sequence],
    outcomes: Sequence,
) -> Dict[str, Dict[str, float]]:
    """
    Rank features by how strongly each tracks the outcome on its own.

    :param features: mapping of feature name to a per-observation series
    :param outcomes: the realised result per observation
    :return: per feature, the Pearson correlation, absolute strength, and rank
    """
    matrix = _as_matrix(features)
    names = list(features)
    y = _check_outcomes(outcomes, matrix.shape[0])
    matrix, y, _ = _clean(matrix, y)

    if y.size < 2:
        raise ValueError('need at least 2 usable observations after cleaning')

    scores = []
    for index, name in enumerate(names):
        scores.append((name, _corr(matrix[:, index], y)))

    strengths = sorted(scores, key=lambda item: abs(item[1]), reverse=True)
    ranking = {name: rank for rank, (name, _) in enumerate(strengths, start=1)}

    return {
        name: {
            'correlation': correlation,
            'abs_correlation': abs(correlation),
            'rank': ranking[name],
        }
        for name, correlation in scores
    }


def ridge_attribution(
    features: Mapping[str, Sequence],
    outcomes: Sequence,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Partition the outcome across features with a ridge fit.

    Features are standardised before fitting so that alpha penalises every
    feature on the same scale; without that, a feature measured in large units
    would dominate the penalty purely through its units.

    :param alpha: ridge strength; larger spreads credit more evenly
    :return: signed contribution per feature, summing with the intercept to the
        fitted value
    """
    if alpha < 0:
        raise ValueError(f'alpha must be non-negative, got {alpha}')

    matrix = _as_matrix(features)
    names = list(features)
    y = _check_outcomes(outcomes, matrix.shape[0])
    matrix, y, _ = _clean(matrix, y)

    if y.size < 2:
        raise ValueError('need at least 2 usable observations after cleaning')

    center = matrix.mean(axis=0)
    spread = matrix.std(axis=0)
    # A constant feature carries no information; leave it at zero rather than
    # dividing by zero and handing it infinite weight.
    scale = np.where(spread > 0, spread, 1.0)
    standardised = (matrix - center) / scale

    y_mean = float(y.mean())
    centred = y - y_mean

    gram = standardised.T @ standardised
    gram += alpha * np.eye(gram.shape[0])
    try:
        coefficients = np.linalg.solve(gram, standardised.T @ centred)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, standardised.T @ centred, rcond=None)[0]

    # Map the standardised coefficient back onto the raw feature scale.
    raw_coefficients = coefficients / scale

    return {name: float(value) for name, value in zip(names, raw_coefficients)}


def attribution_report(
    features: Mapping[str, Sequence],
    outcomes: Sequence,
    alpha: float = 1.0,
    top_k: int = 5,
) -> Dict[str, object]:
    """
    Combine both views into one ranked report.

    :param features: mapping of feature name to a per-observation series
    :param outcomes: the realised result per observation
    :param alpha: ridge strength for the multivariate split
    :param top_k: how many features to name as dominant
    :return: univariate rankings, the ridge split, and a headline summary
    """
    matrix = _as_matrix(features)
    names = list(features)
    y = _check_outcomes(outcomes, matrix.shape[0])
    usable_matrix, usable_y, keep = _clean(matrix, y)

    univariate = univariate_attribution(features, outcomes)
    contributions = ridge_attribution(features, outcomes, alpha=alpha)

    ordered: List[str] = sorted(names, key=lambda n: abs(univariate[n]['correlation']), reverse=True)
    dominant = ordered[: max(1, min(top_k, len(ordered)))]

    explained = 0.0
    if usable_y.size >= 2:
        fitted = np.full(usable_y.shape, float(usable_y.mean()))
        for index, name in enumerate(names):
            fitted += contributions[name] * usable_matrix[:, index]
        total = float(np.sum((usable_y - usable_y.mean()) ** 2))
        explained = float(1.0 - np.sum((usable_y - fitted) ** 2) / total) if total > 0 else 0.0

    return {
        'observations': int(keep.sum()),
        'features': len(names),
        'univariate': univariate,
        'ridge_contributions': contributions,
        'dominant_features': dominant,
        'r_squared_in_sample': explained,
    }
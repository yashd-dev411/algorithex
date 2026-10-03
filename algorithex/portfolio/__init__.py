"""
Aggregator for multi-asset portfolio construction.

Re-exports:
- shrunk_covariance: sample covariance blended towards constant correlation
- equal_weight_weights / minimum_variance_weights: capital-weighted schemes
- risk_parity_weights: weights that equalise each asset's risk contribution
- portfolio_volatility / risk_contributions / diversification_ratio: diagnostics
"""

from .allocation import (
    constant_correlation_covariance,
    diversification_ratio,
    equal_weight_weights,
    minimum_variance_weights,
    portfolio_volatility,
    risk_contributions,
    risk_parity_weights,
    shrunk_covariance,
)

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
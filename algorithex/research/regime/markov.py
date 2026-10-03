"""
Gaussian hidden Markov model.

A small, dependency-free implementation of the standard model: continuous
observations, discrete latent states, diagonal Gaussian emissions. It exists
because knowing *which regime* a market is in is usually more actionable than
knowing a price, and because a regime label is something a strategy can be
conditioned on in a way a moving average is not.

Fitting is Baum-Welch (EM) in log space throughout. Log space is not
optional here: probabilities multiply across thousands of steps and underflow
to zero well before a backtest finishes, which turns a correct implementation
into a silent NaN generator.

The model is diagonal rather than full covariance on purpose. With a few
hundred bars of history, a full covariance matrix is mostly estimation noise
and will happily overfit into states that mean nothing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ['GaussianHMM']

_LOG_2PI = float(np.log(2.0 * np.pi))


def _logsumexp(values: np.ndarray, axis: int = None, keepdims: bool = False):
    """
    log(sum(exp(values))) with the usual max-shift trick.

    SciPy's ``logsumexp`` is well tested but carries enough per-call overhead
    to dominate here: the forward and backward recursions call it twice per
    timestep, so a single EM iteration performs O(T) calls on tiny arrays.
    Fitting a two-state model on a thousand bars took minutes with it and
    seconds without. Same arithmetic, no dispatch cost.

    Matches SciPy's signature, including ``keepdims``.
    """
    values = np.asarray(values, dtype=np.float64)

    if axis is None:
        shifted = np.max(values)
        # An all -inf input would give nan; treat it as contributing nothing.
        if not np.isfinite(shifted):
            shifted = 0.0
        result = float(shifted + np.log(np.sum(np.exp(values - shifted))))
        return np.full((1,) * values.ndim, result) if keepdims else result

    shifted = np.max(values, axis=axis, keepdims=True)
    shifted = np.where(np.isfinite(shifted), shifted, 0.0)
    # A slice that is entirely -inf sums to 0 and takes log to -inf, which is
    # the correct answer (SciPy agrees) but warns on the way. Scoped so genuine
    # numerical warnings elsewhere still surface.
    with np.errstate(divide='ignore'):
        total = shifted + np.log(
            np.sum(np.exp(values - shifted), axis=axis, keepdims=True)
        )
    return total if keepdims else np.squeeze(total, axis=axis)


class GaussianHMM:
    """
    Hidden Markov model with diagonal Gaussian emissions.

    :param n_states: number of latent regimes
    :param n_iter: maximum EM iterations per restart
    :param tol: stop once the log-likelihood improves by less than this
    :param random_state: seed, so a fit is reproducible
    :param n_restarts: independent EM runs; the best likelihood wins. Values
        below about 5 make it likely to return a degenerate solution
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        var_floor: float = 1e-8,
        n_restarts: int = 5,
    ) -> None:
        if n_states < 1:
            raise ValueError(f'n_states must be at least 1, got {n_states}')
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.var_floor = var_floor
        self.n_restarts = n_restarts

        self.startprob_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covars_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.log_likelihood_: Optional[float] = None
        self.n_iter_run_: int = 0

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(f'expected 2D input, got shape {array.shape}')
        if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError('cannot fit on an empty series')
        if not np.all(np.isfinite(array)):
            raise ValueError('input contains NaN or infinite values')
        return array

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log N(x_t | state k) for every t and k. Shape (T, K)."""
        # (T, 1, F) - (1, K, F) -> (T, K, F)
        diff = X[:, None, :] - self.means_[None, :, :]
        exponent = -0.5 * (np.square(diff) / self.covars_[None, :, :])
        # -0.5 * log(2 * pi * variance), i.e. the variance is *logged* inside
        # the normalisation. Adding the raw variance here instead would make a
        # wider state look ever more likely and collapse the fit onto it.
        log_norm = -0.5 * (np.log(self.covars_) + _LOG_2PI)
        return np.sum(exponent + log_norm, axis=2)

    # --------------------------------------------------------- forward/backward

    def _forward(self, log_b: np.ndarray):
        n_samples = log_b.shape[0]
        log_alpha = np.empty((n_samples, self.n_states))
        log_alpha[0] = np.log(self.startprob_) + log_b[0]
        for t in range(1, n_samples):
            log_alpha[t] = _logsumexp(
                log_alpha[t - 1][:, None] + np.log(self.transmat_), axis=0
            ) + log_b[t]
        return log_alpha

    def _backward(self, log_b: np.ndarray):
        n_samples = log_b.shape[0]
        log_beta = np.zeros((n_samples, self.n_states))
        for t in range(n_samples - 2, -1, -1):
            log_beta[t] = _logsumexp(
                np.log(self.transmat_) + (log_b[t + 1] + log_beta[t + 1])[None, :],
                axis=1,
            )
        return log_beta

    # -------------------------------------------------------------------- fit

    def _initialise(self, X: np.ndarray, rng: np.random.Generator, strategy: int) -> None:
        """
        Seed the parameters.

        Seeding every state at the global mean and global variance looks
        neutral but is fatal: EM is symmetric under permuted states, so two
        identically-initialised states stay identical forever and the model
        collapses to a single Gaussian.

        Two seeding strategies are used across restarts because neither
        dominates. Quantile blocks separate regimes that differ in *level*
        (a trending versus ranging market). Random spread separates regimes
        that differ in *volatility* while sharing a level, which is the common
        case and the one quantile seeding handles worst.
        """
        n_features = X.shape[1]

        if strategy == 0:
            order = np.argsort(X[:, 0], kind='stable')
            blocks = np.array_split(order, self.n_states)
            means = np.empty((self.n_states, n_features))
            variances = np.empty((self.n_states, n_features))
            for state, block in enumerate(blocks):
                if block.size == 0:
                    means[state] = X.mean(axis=0)
                    variances[state] = X.var(axis=0)
                else:
                    means[state] = X[block].mean(axis=0)
                    variances[state] = X[block].var(axis=0)
        else:
            global_mean = X.mean(axis=0)
            global_var = np.maximum(X.var(axis=0), self.var_floor)
            global_std = np.sqrt(global_var)
            means = global_mean + rng.normal(0.0, 0.6, (self.n_states, n_features)) * global_std
            scales = np.exp(rng.normal(0.0, 1.1, (self.n_states, n_features)))
            variances = global_var * scales

        self.means_ = means
        self.covars_ = np.maximum(variances, self.var_floor)

        # Persistent states: real regimes last longer than a bar, so start from
        # a high self-transition prior instead of a uniform one.
        diagonal = 0.8 if self.n_states > 1 else 1.0
        off = (1.0 - diagonal) / max(1, self.n_states - 1)
        self.transmat_ = np.full((self.n_states, self.n_states), off)
        np.fill_diagonal(self.transmat_, diagonal)
        self.transmat_ += rng.normal(0.0, 1e-3, self.transmat_.shape)

        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.startprob_ += rng.normal(0.0, 1e-3, self.startprob_.shape)

    def _normalise(self, array: np.ndarray) -> np.ndarray:
        total = array.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(array.shape, 1.0 / array.size)
        return array / total

    def _e_step(self, log_b: np.ndarray):
        log_alpha = self._forward(log_b)
        log_beta = self._backward(log_b)
        log_likelihood = float(_logsumexp(log_alpha[-1]))

        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        # Renormalise each step: log-space rounding can drift over long series.
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

        xi = np.zeros((self.n_states, self.n_states))
        for t in range(log_b.shape[0] - 1):
            log_xi = (
                log_alpha[t][:, None]
                + np.log(self.transmat_)
                + log_b[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - log_likelihood
            )
            xi += np.exp(log_xi)

        return gamma, xi, log_likelihood

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        self.startprob_ = self._normalise(gamma[0])

        expected_transitions = np.maximum(xi.sum(axis=1), 1e-300)
        self.transmat_ = xi / expected_transitions[:, None]
        self.transmat_ = np.maximum(self.transmat_, 1e-300)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        expected_states = np.maximum(gamma.sum(axis=0), 1e-300)
        self.means_ = (gamma.T @ X) / expected_states[:, None]
        variance = (gamma.T @ np.square(X)) / expected_states[:, None]
        self.covars_ = np.maximum(
            variance - np.square(self.means_), self.var_floor
        )

    def fit(self, X):
        """
        Fit the model with Baum-Welch, keeping the best of several restarts.

        EM is not convex and a single run is a coin flip: a poor seed converges
        to a degenerate solution where one state swallows the whole series and
        the Viterbi path never changes state. Each restart explores a different
        basin and the highest-likelihood parameters win, which is the standard
        remedy and the reason this is not just one ``_initialise`` call.

        :param X: (T, F) observations; a 1D array is read as a single feature
        :return: self, so calls can be chained
        """
        X = self._as_2d(X)
        self.n_features_ = X.shape[1]

        best = None
        best_likelihood = -np.inf
        for restart in range(max(1, self.n_restarts)):
            rng = np.random.default_rng(
                None if self.random_state is None else self.random_state + restart
            )
            self._initialise(X, rng, strategy=restart % 2)
            likelihood = self._run_em(X)
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best = (
                    self.startprob_.copy(),
                    self.transmat_.copy(),
                    self.means_.copy(),
                    self.covars_.copy(),
                    self.n_iter_run_,
                )

        self.startprob_, self.transmat_, self.means_, self.covars_, self.n_iter_run_ = best
        self.log_likelihood_ = best_likelihood
        return self

    def _run_em(self, X: np.ndarray) -> float:
        """One Baum-Welch run from the current parameters. Returns the likelihood."""
        previous = -np.inf
        likelihood = -np.inf
        for iteration in range(self.n_iter):
            gamma, xi, likelihood = self._e_step(self._log_emission(X))
            self._m_step(X, gamma, xi)
            self.n_iter_run_ = iteration + 1
            if np.isfinite(previous) and likelihood - previous < self.tol:
                break
            previous = likelihood

        return float(_logsumexp(self._forward(self._log_emission(X))[-1]))

    # -------------------------------------------------------------- inference

    def _viterbi(self, log_b: np.ndarray) -> np.ndarray:
        n_samples = log_b.shape[0]
        delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)

        delta[0] = np.log(self.startprob_) + log_b[0]
        for t in range(1, n_samples):
            candidates = delta[t - 1][:, None] + np.log(self.transmat_)
            psi[t] = np.argmax(candidates, axis=0)
            delta[t] = candidates[psi[t], np.arange(self.n_states)] + log_b[t]

        path = np.zeros(n_samples, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(n_samples - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def predict(self, X) -> np.ndarray:
        """Most likely state at each step, via Viterbi decoding."""
        X = self._as_2d(X)
        self._require_fitted()
        return self._viterbi(self._log_emission(X))

    def predict_proba(self, X) -> np.ndarray:
        """Posterior probability of each state at each step. Shape (T, K)."""
        X = self._as_2d(X)
        self._require_fitted()
        log_b = self._log_emission(X)
        log_gamma = self._forward(log_b) + self._backward(log_b)
        posterior = np.exp(log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True))
        return posterior

    def score(self, X) -> float:
        """Log-likelihood of a sequence under the fitted model."""
        X = self._as_2d(X)
        self._require_fitted()
        return float(_logsumexp(self._forward(self._log_emission(X))[-1]))

    def _require_fitted(self) -> None:
        if self.means_ is None:
            raise RuntimeError('model must be fitted before use')
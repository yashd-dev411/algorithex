"""
Aggregator for research auditing.

Re-exports:
- detect_lookahead / audit_source_file: static AST analysis of strategy source
- CandleView / LeakageGuard: runtime detection of future-bar reads
- audit_robustness / certify_robustness: adversarial falsification of results
- adversarial_path: worst-case price path inside a perturbation budget
- cost_stress_curve / break_even_multiple: how much cost kills the edge
"""

from .leakage import (
    AuditReport,
    CandleView,
    Finding,
    LeakageGuard,
    audit_source_file,
    detect_lookahead,
)
from .robustness import (
    DEFAULT_ATTACKS,
    Attack,
    AttackResult,
    RobustnessReport,
    adversarial_path,
    apply_attack,
    audit_robustness,
    block_bootstrap,
    break_even_multiple,
    certify_robustness,
    cost_stress_curve,
    drop_random_bars,
    inject_spikes,
    jitter_prices,
    perturbation_sensitivity,
    pnl_retention,
    scramble_volumes,
    shuffle_returns,
    survival_rate,
)

__all__ = [
    'Finding',
    'AuditReport',
    'detect_lookahead',
    'audit_source_file',
    'CandleView',
    'LeakageGuard',
    'Attack',
    'AttackResult',
    'RobustnessReport',
    'DEFAULT_ATTACKS',
    'jitter_prices',
    'shuffle_returns',
    'scramble_volumes',
    'inject_spikes',
    'block_bootstrap',
    'drop_random_bars',
    'apply_attack',
    'survival_rate',
    'pnl_retention',
    'cost_stress_curve',
    'break_even_multiple',
    'perturbation_sensitivity',
    'adversarial_path',
    'certify_robustness',
    'audit_robustness',
]
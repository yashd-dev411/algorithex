"""
Lookahead and leakage detection.

A backtest that peeks at the future produces a beautiful equity curve and a
live account that loses money. It is the most common way a trading system fails
and the most expensive, because nothing about the result looks wrong.

The failure has two faces, and this module covers both.

Static analysis
    Parse the strategy source and flag constructs that *cannot* be
    lookahead-free: negative indexing into the current bar's series, a negative
    ``shift``, ``center=True`` on a rolling window, a backwards ``np.roll``.
    These are decidable from the source alone, so they can be caught in CI
    before a backtest is ever run.

Runtime tracking
    Static analysis cannot see data-dependent indexing, so
    :class:`CandleView` wraps the candle array and records the highest index
    the strategy actually touched. When that index runs ahead of the bar being
    processed, the strategy has read something it could not have known.

The distinction between the two matters. Static findings are certain. Runtime
findings are certain too, but only for the paths a backtest happened to
exercise, so absence of runtime findings is weaker evidence than presence.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    'Finding',
    'AuditReport',
    'detect_lookahead',
    'audit_source_file',
    'CandleView',
    'LeakageGuard',
]

# Names that conventionally hold the current bar's data. Negative indexing into
# one of these is almost always a lookahead.
_SERIES_HINTS = ('candle', 'candles', 'price', 'prices', 'close', 'closes', 'high',
                 'low', 'open', 'volume', 'data', 'series', 'values', 'arr')


@dataclass(frozen=True)
class Finding:
    """One suspicious construct, located precisely enough to jump to."""

    line: int
    column: int
    rule: str
    message: str
    severity: str = 'error'
    snippet: str = ''

    def __str__(self) -> str:
        return f'{self.severity.upper()} line {self.line}: {self.message}'


@dataclass
class AuditReport:
    """Everything a static audit found."""

    findings: List[Finding]
    source_name: str = '<source>'
    lines_examined: int = 0

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == 'error']

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == 'warning']

    @property
    def clean(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.clean

    def summary(self) -> str:
        if self.clean:
            return f'{self.source_name}: clean ({self.lines_examined} lines)'
        return (
            f'{self.source_name}: {len(self.errors)} error(s), '
            f'{len(self.warnings)} warning(s)'
        )


def _is_series_name(node: ast.AST) -> bool:
    """
    Whether a node looks like the current bar's data.

    Checks the attribute name as well as the object it hangs off, because the
    common case is ``self.candles[-1]`` where the informative name is on the
    attribute and the base is just ``self``.
    """
    if isinstance(node, ast.Name):
        name = node.id.lower()
        return any(hint in name for hint in _SERIES_HINTS)
    if isinstance(node, ast.Attribute):
        if any(hint in node.attr.lower() for hint in _SERIES_HINTS):
            return True
        return _is_series_name(node.value)
    return False


def _source_segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    index = getattr(node, 'lineno', 1) - 1
    if 0 <= index < len(lines):
        return lines[index].strip()
    return ''


class _LookaheadVisitor(ast.NodeVisitor):
    """Collects constructs that read data the strategy could not have had."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.findings: List[Finding] = []

    def _add(self, node: ast.AST, rule: str, message: str, severity: str = 'error') -> None:
        self.findings.append(
            Finding(
                line=getattr(node, 'lineno', 0),
                column=getattr(node, 'col_offset', 0),
                rule=rule,
                message=message,
                severity=severity,
                snippet=_source_segment(self.source, node),
            )
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # candles[-1] and friends: the final bar is the future from inside a
        # live strategy, and the current bar's last element in a backtest.
        if isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub):
            if _is_series_name(node.value):
                self._add(
                    node,
                    'negative-index',
                    'negative indexing reads from the end of the series, which is '
                    'the future relative to the bar being processed',
                )
        elif isinstance(node.slice, ast.Slice) and node.slice.step is None:
            # In the AST a slice's fields are lower/upper; start/stop are the
            # builtin's names and do not exist here.
            upper = node.slice.upper
            if upper is None or (
                isinstance(upper, ast.UnaryOp) and isinstance(upper.op, ast.USub)
            ):
                if _is_series_name(node.value):
                    self._add(
                        node,
                        'open-ended-slice',
                        'slice with no explicit end reads to the end of the series, '
                        'including bars that have not happened yet',
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)

        # pandas .shift(-1) is the future, unambiguously.
        if name.endswith('.shift') and node.args:
            first = node.args[0]
            if isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub):
                self._add(
                    node, 'negative-shift',
                    'shift(-n) moves values forward in time, importing the future',
                )
        for keyword in node.keywords:
            if keyword.arg == 'shift' and isinstance(keyword.value, ast.UnaryOp):
                if isinstance(keyword.value.op, ast.USub):
                    self._add(
                        node, 'negative-shift',
                        'shift=-n moves values forward in time, importing the future',
                    )
            # A centred rolling window consumes half the window from the future.
            if keyword.arg == 'center' and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    self._add(
                        node, 'centered-window',
                        'center=True makes the window use bars on both sides, '
                        'including future ones',
                    )

        if name in ('np.roll', 'roll') and node.args and len(node.args) >= 2:
            shift = node.args[1]
            if isinstance(shift, ast.UnaryOp) and isinstance(shift.op, ast.USub):
                self._add(
                    node, 'negative-roll',
                    'roll(x, -n) shifts values backward in index, importing the future',
                )

        self.generic_visit(node)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f'{base}.{func.attr}' if base else func.attr
    return ''


def detect_lookahead(source: str, source_name: str = '<source>') -> AuditReport:
    """
    Statically scan source for constructs that introduce lookahead.

    :param source: Python source to analyse
    :param source_name: label used in the report
    :return: every finding, with line numbers
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return AuditReport(
            findings=[Finding(
                line=exc.lineno or 0, column=exc.offset or 0, rule='syntax-error',
                message=f'could not parse source: {exc.msg}', severity='error',
            )],
            source_name=source_name,
        )

    visitor = _LookaheadVisitor(source)
    visitor.visit(tree)

    findings = sorted(visitor.findings, key=lambda f: (f.line, f.column))
    return AuditReport(
        findings=findings,
        source_name=source_name,
        lines_examined=len(source.splitlines()),
    )


def audit_source_file(path, source_name: Optional[str] = None) -> AuditReport:
    """Run :func:`detect_lookahead` against a file on disk."""
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    return detect_lookahead(source, source_name or str(path))


class CandleView:
    """
    A read-only window onto candle data that records the furthest index touched.

    Wrap the series a strategy sees and the leakage becomes measurable rather
    than a matter of opinion: if ``max_index_read`` exceeds the bar being
    processed, the strategy looked ahead.

    :param data: the underlying series
    :param name: label used in error messages
    """

    def __init__(self, data, name: str = 'candles') -> None:
        self._data = np.asarray(data)
        self._name = name
        self.max_index_read: int = -1
        self.reads: List[int] = []

    def _record(self, index: int) -> None:
        self.max_index_read = max(self.max_index_read, index)
        self.reads.append(index)

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __getitem__(self, key):
        if isinstance(key, int):
            self._record(key)
        elif isinstance(key, slice):
            if key.start is not None:
                self._record(key.start)
            if key.stop is not None:
                self._record(key.stop - 1)
            elif key.start is None and key.stop is None:
                self._record(len(self) - 1)
        return self._data[key]

    @property
    def underlying(self):
        """The raw array, for callers that genuinely need it."""
        return self._data


class LeakageGuard:
    """
    Tracks whether a strategy read ahead of the bar it was processing.

    :param bar_index: index of the bar currently being processed
    """

    def __init__(self, bar_index: int = 0) -> None:
        self.bar_index = bar_index
        self.violations: List[dict] = []
        self._tracked: List[tuple] = []

    def track(self, view: CandleView, label: str = '') -> None:
        """Register a view so it is checked whenever :meth:`check` runs."""
        self._tracked.append((view, label))

    def set_bar(self, index: int) -> None:
        self.bar_index = index

    def check(self) -> List[dict]:
        """
        Compare every tracked view's highest read against the current bar.

        :return: one record per view that read ahead; empty when clean
        """
        violations: List[dict] = []
        for view, label in self._tracked:
            if view.max_index_read > self.bar_index:
                record = {
                    'source': label or view._name,
                    'bar': self.bar_index,
                    'max_index_read': view.max_index_read,
                    'bars_ahead': view.max_index_read - self.bar_index,
                }
                violations.append(record)
                self.violations.append(record)
        return violations

    def reset_reads(self) -> None:
        """Clear the per-bar read record, keeping the bar index."""
        for view, _label in self._tracked:
            view.max_index_read = -1
            view.reads.clear()

    @property
    def clean(self) -> bool:
        return not self.violations
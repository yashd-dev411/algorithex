<div align="center">
<br>
<p align="center">
<img src="assets/algorithex-logo.svg" alt="Algorithex" height="76" />
</p>

<p align="center">
Backtest · Optimize · Validate · Trade — entirely on your own machine.
</p>
</div>

# Algorithex

> **This is a fork of [Jesse](https://github.com/jesse-ai/jesse)**, the open-source crypto trading framework by Jesse Mir and contributors, released under the MIT License.
>
> Upstream is credited in [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). This fork rebrands the package to `algorithex`, removes the hosted-service dependencies so it runs fully self-hosted, and adds four new research modules described under [Research tooling](#research-tooling). All local branding, packaging, and Docker names use Algorithex.

A local, self-hosted trading framework. The Python package is `algorithex`.

---

Algorithex is an advanced crypto trading framework that aims to **simplify** **researching** and defining **YOUR OWN trading strategies** for backtesting, optimizing, and paper trading — fully local, no website account needed.

## What is Algorithex?
Algorithex runs the Jesse research engine locally under your own brand: backtests, optimization, Monte Carlo, and rule significance tests without redirects or license tokens.

## Why Algorithex?
In short, Algorithex is more **accurate** than other solutions, and way more **simple**.
In fact, it is so simple that in case you already know Python, you can get started today, in **matter of minutes**, instead of **weeks and months**. 

## Key Features

- 📝 **Simple Syntax**: Define both simple and advanced trading strategies with the simplest syntax in the fastest time.
- 📊 **Comprehensive Indicator Library**: Access a complete library of technical indicators with easy-to-use syntax.
- 📈 **Smart Ordering**: Supports market, limit, and stop orders, automatically choosing the best one for you.
- ⏰ **Multiple Timeframes and Symbols**: Backtest and livetrade multiple timeframes and symbols simultaneously without look-ahead bias.
- 🔒 **Self-Hosted and Privacy-First**: Designed with your privacy in mind, fully self-hosted to ensure your trading strategies and data remain secure.
- 🛡️ **Risk Management**: Built-in helper functions for robust risk management.
- 📋 **Metrics System**: A comprehensive metrics system to evaluate your trading strategy's performance.
- 🔍 **Debug Mode**: Observe your strategy in action with a detailed debug mode.
- 🔧 **Optimize Mode**: Search strategy parameters efficiently with Optuna and parallel processing powered by Ray.
- 📈 **Leveraged and Short-Selling**: First-class support for leveraged trading and short-selling.
- 🔀 **Partial Fills**: Supports entering and exiting positions in multiple orders, allowing for greater flexibility.
- 🔔 **Advanced Alerts**: Create real-time alerts within your strategies for effective monitoring.
- 🔌 **Algorithex MCP**: Connect Claude, Codex, Cursor, VS Code, Zed, and other MCP-compatible AI assistants directly to your local Algorithex project.
- 🔧 **Built-in Code Editor**: Write, edit, and debug your strategies with a built-in code editor.
- 📊 **Interactive Trading Charts**: Inspect candles, strategy indicators, horizontal levels, orders, and completed trades across backtest, paper, and live sessions.
- 🔬 **Rule Significance Testing**: Test whether an entry rule shows genuine historical edge or could have appeared by chance.
- 🎲 **Monte Carlo Analysis**: Stress-test your strategies with trade-order shuffling and candles-based simulations to distinguish skill from luck and guard against overfitting.
- 🧠 **Machine Learning**: A built-in ML pipeline — gather labelled training data from backtests, train scikit-learn models (binary, multiclass, or regression), and deploy predictions directly inside your strategies.
- 🧪 **Research API and Jupyter**: Run backtests, optimization, significance tests, Monte Carlo analysis, candle workflows, and machine learning from Python scripts or notebooks.
- 🦀 **Rust-Powered Indicators**: Native Rust implementations make indicator-heavy strategies and large research runs substantially faster.
- 🤖 **Reinforcement Learning — Coming Soon**: First-class reinforcement-learning workflows built on Algorithex's simulation and research stack are on the way.
- 📺 **Youtube Channel**: Algorithex has a Youtube channel with screencast tutorials that go through example strategies step by step.

## Dive Deeper into Algorithex's Capabilities

### Stupid Simple
Craft complex trading strategies with remarkably simple Python. Access 300+ indicators, multi-symbol/timeframe support, spot/futures trading, partial fills, and risk management tools. Focus on logic, not boilerplate.

```python
class GoldenCross(Strategy):
    def should_long(self):
        # go long when the EMA 8 is above the EMA 21
        short_ema = ta.ema(self.candles, 8)
        long_ema = ta.ema(self.candles, 21)
        return short_ema > long_ema

    def go_long(self):
        entry_price = self.price - 10        # limit buy order at $10 below the current price
        qty = utils.size_to_qty(self.balance*0.05, entry_price) # spend only 5% of my total capital
        self.buy = qty, entry_price                 # submit entry order
        self.take_profit = qty, entry_price*1.2  # take profit at 20% above the entry price
        self.stop_loss = qty, entry_price*0.9   # stop loss at 10% below the entry price
```

### Backtest
Execute highly accurate and fast backtests without look-ahead bias. Utilize debugging logs, interactive charts with indicator support, and detailed performance metrics to validate your strategies thoroughly.

### Interactive Trading Charts
Inspect your strategy where its decisions happened. Algorithex combines candlesticks, strategy-added indicators and levels, executed orders, and completed trades in synchronized interactive charts. The same charting workflow is available for backtests and for running or completed paper/live sessions.


Expand a trade to inspect every execution, collapse or isolate indicator panes, follow OHLC and indicator values under the cursor, reset the view, use fullscreen mode, or export the chart as an image.

### Live/Paper Trading
Deploy strategies live with robust monitoring tools. Supports paper trading, multiple accounts, real-time logs & notifications (Telegram, Slack, Discord), interactive charts, spot/futures, DEX, and a built-in code editor.

### Benchmark
Accelerate research using the benchmark feature. Run batch backtests, compare across timeframes, symbols, and strategies. Filter and sort results by key performance metrics for efficient analysis.

### Algorithex MCP: Your AI Assistant, Connected to Algorithex
Algorithex includes a local Model Context Protocol (MCP) server. Connect your preferred AI assistant and let it work with Algorithex's real tools and project context instead of merely guessing how your trading framework behaves.

Through Algorithex MCP, an assistant can help you write and improve strategies, manage candle data, run and inspect backtests, perform rule significance tests, optimize parameters, run Monte Carlo simulations, and link you directly to the saved results in the Algorithex dashboard. Your strategies and data remain under your control in your self-hosted Algorithex setup.

For example, you can ask:

> Check whether my new entry rule is statistically significant, backtest it, optimize the promising parameters, and run a candles-based Monte Carlo analysis before we consider paper trading.

Connect Claude, Codex, Cursor, VS Code, or Zed to Algorithex →

### Rule Significance Testing
Before spending hours building and tuning a complete strategy, test whether its entry rule has a measurable historical edge. Algorithex compares the rule against a bootstrap distribution of random entries on the same market history, helping you reject noisy ideas early and focus your research on signals worth developing.

Learn about Rule Significance Testing →

### Monte Carlo Analysis
Stress-test your strategies beyond a single historical path. Algorithex's Monte Carlo mode runs hundreds of simulations using **trade-order shuffling** (tests whether trade timing drove your results) and **candles-based** (tests robustness under slightly different market conditions) methods. Use it to distinguish skill from luck, understand the range of outcomes you can realistically expect, and catch overfitting early.

### Research tooling

> **New in this fork.** These modules are original work and are not part of upstream Jesse.

They are importable libraries rather than CLI commands, and they are pure computation — no database or engine state — so they can drive a backtest, a live run, or a plain NumPy array.

#### Walk-forward validation — `algorithex.research.walkforward`

The project could optimise but had no discipline for scoring results **out of sample**. This adds it.

```python
from algorithex.research.walkforward import purged_walk_forward, evaluate_windows

windows = purged_walk_forward(n_obs, train_size=2000, test_size=250, embargo=48)
report = evaluate_windows(windows, per_bar_returns)

print(report["stitched"]["sharpe"])
print(report["spread"]["sharpe"])          # min / median / max across paths
print(report["profitable_path_ratio"])     # how many paths were actually positive
```

- **Rolling and anchored** windows: expanding history or fixed-length.
- **Purging and embargo**: a bar's outcome is not known at that bar. Training observations whose label window reaches into the test set are dropped, so the fit cannot see the answer it is scored against. This is the single most important parameter in the module — set it to 0 and lookahead bias returns.
- **Combinatorial purged cross-validation**: split into *N* contiguous blocks, choose combinations of *k* as test sets, train on the purged remainder. 6-choose-2 gives 15 paths through the same data, exposing results that depend on one lucky partition.
- `stitch_oos_returns` collapses overlapping test windows so each bar is counted once, and `evaluate_windows` reports the *distribution* of results across paths rather than the maximum.

#### Regime detection — `algorithex.research.regime`

```python
from algorithex.research.regime import GaussianHMM, detect_change_points, regime_position_size

model = GaussianHMM(n_states=3, random_state=0).fit(features)   # NumPy, no hmmlearn needed
regime = model.predict(features)                                # Viterbi decode
breakpoints = detect_change_points(close_prices, min_segment=50)
size = regime_position_size(regime, returns, target_vol=0.01)   # calmer regime -> larger size
```

- **Gaussian HMM** with Baum-Welch fitting in log space and Viterbi decoding, implemented directly on NumPy/SciPy. Verified against brute-force enumeration of every state path, so the likelihood is exact rather than merely plausible. Uses multiple restarts because a single EM run frequently collapses onto one degenerate state.
- **Change-point detection** by binary segmentation with a BIC-style penalty. Recovers clean level shifts while reporting none on stationary noise.
- **Adaptive sizing**: volatility targeting, per-regime sizing, and drawdown throttling that restores size as drawdown heals rather than staying permanently throttled by an old high.

#### Multi-asset portfolio engine — `algorithex.portfolio`

```python
from algorithex.portfolio import (
    shrunk_covariance, risk_parity_weights, risk_contributions, portfolio_volatility,
)

cov = shrunk_covariance(returns_matrix, shrinkage=0.3)   # (T, N) returns
weights = risk_parity_weights(cov)
print(risk_contributions(weights, cov) / portfolio_volatility(weights, cov))  # equal shares
```

- Equal weight, minimum variance, and **risk parity** — weights that equalise each asset's *risk* rather than its capital.
- Covariance is **shrunk** towards a constant-correlation target, because a sample covariance from few observations of many assets is singular and optimisers will happily exploit that instability.
- `risk_contributions` exists so you can *check* that risk parity did its job rather than trusting it.
- Weight caps are supported and verified to bind when they should.

#### Attribution — `algorithex.research.attribution`

```python
from algorithex.research.attribution import attribute_trades, attribution_report

report = attribute_trades(trades)                       # dicts or model instances
print(report["by_symbol"], report["profit_concentration"])

features = attribution_report(feature_series, outcomes)  # which indicators earned the PnL
print(features["dominant_features"], features["r_squared_in_sample"])
```

- **Trade-level**: PnL by symbol, side and time bucket, plus `profit_concentration` — the share of gross profit carried by your single best trade. Above roughly 0.3 is the number worth arguing with.
- **Execution quality**: compares captured profit against the favourable excursion actually available while the trade was open. Returns `available=False` rather than a fabricated zero when excursion data was not supplied.
- **Feature-level**: univariate correlation plus a ridge split that partitions credit across correlated features. Ridge rather than least squares because indicator features are heavily collinear and the unregularised fit hands the whole edge to whichever pair happens to be most correlated.

#### Adversarial robustness — `algorithex.audit.robustness`

A backtest says what happened. This asks whether it would survive being **slightly wrong** — and then tries to break it on purpose.

```python
from algorithex.audit.robustness import audit_robustness, certify_robustness

report = audit_robustness(lambda prices: pnl_of(strategy, prices), close_prices)
print(report.verdict)          # ROBUST or FRAGILE
print(report.survival_rate)    # how much profit survives the attacks
print(report.break_even_cost)  # cost multiplier at which PnL hits zero

# Not a checklist — a certificate. Searches for the worst price path.
cert = certify_robustness(fn, close_prices, budgets=(0.01, 0.02, 0.05))
print(cert["margin_of_safety"])   # smallest perturbation that could have produced a loss
```

- **Six attacks on the data**: proportional jitter, return shuffling (keeps the distribution, destroys the order), parametric resampling, single-bar price spikes, block bootstrap, and dropped bars.
- **Certified worst case** via projected gradient ascent on the log-price path. Unlike sampling attacks, this searches the whole ±budget box, so the answer is "no price path this far from the historical one could have lost money" — and it reports the budget at which that stops being true.
- **Break-even cost** by bisection, not by assuming a curve shape.
- Every step is seeded, so a report is reproducible.

#### Strategy ensembles — `algorithex.research.ensemble`

Running one strategy is the wrong frame. These decide **how much of each**, given what is known now.

```python
from algorithex.research.ensemble import StrategyReturns, walk_forward_ensemble, evaluate_ensemble

members = [StrategyReturns(f"s{i}", r) for i, r in enumerate(strategy_returns)]
path = walk_forward_ensemble(matrix, scheme="inverse_volatility", train_window=252)
report = evaluate_ensemble(members, weight_path=path.matrix)

print(report.effective_strategies)  # 1.0 == "your ensemble is really one trade"
print(report.harmful_members())      # whose removal would improve the Sharpe
```

- **Allocation schemes**: equal weight, inverse volatility, exponential (multiplicative-weights, regret-bounded), and drawdown-penalised.
- **Information-ratio arbitration** sizes on *conditional* performance, which inverse volatility structurally cannot do — it equalises risk but cannot tell a winner from a loser.
- **Regime-conditional weights**, shrunk towards the global allocation and refused below a minimum observation count, because per-regime stacking on thin evidence is how alphas get invented.
- **Causal by construction**: `walk_forward_ensemble` refits on a trailing window only, and reports turnover so a churning allocator cannot hide its costs.
- `marginal_contribution` measures what each member actually adds, which is invisible in per-strategy Sharpe ratios.

#### Backtest overfitting — `algorithex.research.overfitting`

Your Sharpe is inflated by however many parameter sets you tried before this one looked good.

```python
from algorithex.research.overfitting import probability_of_backtest_overfitting, deflated_sharpe

pbo = probability_of_backtest_overfitting(all_configuration_returns, n_blocks=8)
print(pbo.pbo, pbo.verdict)          # probability the in-sample winner was not the OOS winner

d = deflated_sharpe(best_returns, n_trials=len(configurations_you_tried))
print(d.deflated_sharpe, d.verdict)   # the Sharpe you should have believed
```

- **PBO by CSCV**: every way of splitting contiguous blocks in half, asking where the in-sample best lands out-of-sample. Pure noise gives ≈ 0.5; a real edge gives ≈ 0.
- **Deflated Sharpe** subtracts the expected maximum Sharpe from `N` trials, using the Euler-Mascheroni correction that makes the estimate honest. Pass the number of trials that *failed* too — that is the whole point.
- **Plateau analysis** separates a genuine effect from a fitted spike: a real signal keeps working when its parameters are nudged.

#### Scheduled execution — `algorithex.execution.algos`

Jesse fills an order at the next candle's close, minus a percentage. Real size does not print all at once.

```python
from algorithex.execution.algos import ExecutionParams, is_optimal, compare_algorithms, simulate_execution

params = ExecutionParams(total_qty=100, volatility=0.4, impact=50.0, risk_aversion=1.0)
plan = is_optimal(params, bar_volumes)       # Almgren-Chriss, solved for bars
result = simulate_execution(plan, closes, volumes, impact=50.0, spread_cost=0.2)

print(result.timing_cost, result.impact_cost, result.spread_cost)
compare_algorithms(params, closes, volumes).best()
```

- **TWAP, POV, VWAP** and **Almgren-Chriss optimal** trajectory. The optimum is a function of the ratio between timing risk and market impact — it front-loads when volatility is high and back-loads when impact is expensive, and is dominated by neither naive rule.
- **Adaptive POV** with a live implementation-shortfall feedback loop that speeds up when it falls behind the volume benchmark.
- **Shortfall decomposition**: timing, spread and impact are reported separately, because they call for opposite responses.
- Participation ceilings are enforced honestly — an order the tape cannot absorb reports itself as partially filled rather than silently shrinking.

#### Certification — `algorithex.research.certification`

One grade, one verdict, and the reasoning behind both.

```python
from algorithex.research.certification import certify

report = certify(
    strategy_name="my-strategy",
    source_code=open("strategies/MyStrategy/__init__.py").read(),
    strategy=lambda prices: pnl_of(strategy, prices),
    prices=close_prices,
    configuration_returns=all_configurations,   # every parameter set you tried
    best_returns=best_configuration,
    window_results=walk_forward_report,
)
print(report.render())
```

Composes the leakage audit, the overfitting checks, the robustness attacks, cost survival and walk-forward consistency into a single graded decision.

- **Missing evidence is not a pass.** A check you supplied no input for is `not_run` and counts against the grade — dropping a check can never improve the score.
- **One blocking failure is enough.** Look-ahead or a strategy that loses money caps the grade at F outright, so nothing averages its way past cheating.

From the command line:

```bash
algorithex certify --strategy strategies/MyStrategy/__init__.py
algorithex certify --strategy strategies/MyStrategy/__init__.py --prices prices.csv
```

The source audit always runs. Adding `--prices` (a CSV with a `close` column) is
enough to score the strategy end to end in Python. The command exits non-zero
unless the report certifies, so it works as a CI gate.

### Machine Learning
Algorithex includes a complete, end-to-end ML pipeline built for trading strategies:

1. **Gather data** — run a backtest in gather mode; call `record_features({...})` at each signal bar and `record_label(name, value)` when the outcome is known. Data is auto-saved to CSV.
2. **Train a model** — call `train_model()` with any scikit-learn–compatible estimator and choose a task type: `"binary"` classification, `"multiclass"` classification, or `"regression"`. Get a full report with feature importance, calibration, and metrics.
3. **Deploy** — switch to deploy mode and call `ml_predict()` or `ml_predict_proba()` inside your strategy. Model loading, scaling, and feature ordering are handled automatically.

```python
# Gather phase — inside your strategy
def before(self):
    self.record_features({
        'rsi': ta.rsi(self.candles),
        'adx': ta.adx(self.candles),
    })

# Deploy phase — gate entries with model confidence
def should_long(self):
    proba = self.ml_predict_proba()
    return proba['long'] > 0.65
```

Explore Algorithex's machine-learning pipeline →

### Research API and Jupyter Notebooks
Everything does not have to happen through the dashboard. Algorithex's Research API exposes candle management, backtesting, optimization, Rule Significance Testing, Monte Carlo analysis, indicators, and machine learning to ordinary Python scripts and Jupyter notebooks. Use it for reproducible experiments, custom reports, batch research, or integration with your existing data-science workflow.

Explore the Research API →

### Rust-Powered Performance
Algorithex's indicators are powered by native Rust, making them significantly faster than common alternatives such as TA-Lib.

The kernel is built from source in [`native-kernel/`](native-kernel) and the compiled binaries are **vendored**, so installing this package pulls in no external native dependency. Prebuilt binaries ship for Linux, Windows and macOS (Apple Silicon, `aarch64`, macOS 11.0 and later) and are built against the Python stable ABI, so one binary per platform serves CPython 3.10 and above. If you want to rebuild the macOS binary yourself, [`native-kernel/build-macos.sh`](native-kernel/build-macos.sh) cross-compiles from any host — no Mac or Xcode required.

### Reinforcement Learning — Coming Soon
We are working on first-class reinforcement-learning support built on Algorithex's simulation and research stack. The goal is to make training, evaluating, and deploying reinforcement-learning agents feel as integrated as Algorithex's existing backtesting, optimization, Monte Carlo, and machine-learning workflows.

### Optimize Your Strategies
Unsure about optimal parameters? Let the optimization mode decide using simple syntax. Fine-tune any strategy parameter with the Optuna library and easy cross-validation.

```python
@property
def slow_sma(self):
    return ta.sma(self.candles, self.hp['slow_sma_period'])

@property
def fast_sma(self):
    return ta.sma(self.candles, self.hp['fast_sma_period'])

def hyperparameters(self):
    return [
        {'name': 'slow_sma_period', 'type': int, 'min': 150, 'max': 210, 'default': 200},
        {'name': 'fast_sma_period', 'type': int, 'min': 20, 'max': 100, 'default': 50},
    ]
```

## Getting Started

Everything runs locally — no account and no website required:

```bash
cd deploy
cp .env.example .env
docker compose up -d
```

Then open <http://localhost:9000>. The dashboard password is `PASSWORD` in
`deploy/.env`. The MCP server is at <http://localhost:9002/mcp>; see
[AGENTS.md](AGENTS.md) for the strategy-authoring workflow it exposes.

## Deploying

Everything needed to run Algorithex is in this repository. Clone it, and you
have a deployable stack.

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>/deploy
cp .env.example .env          # then edit it, see below
docker compose up -d
docker compose ps             # wait until `algorithex` reads healthy
```

The dashboard is on <http://localhost:9000>. Sign in with the `PASSWORD` from
`.env`.

`docker compose ps` is the thing to watch on first run. The app is gated on
Postgres and Redis passing their healthchecks before it starts, so you should see
them report `healthy` and *then* the app start. The first boot is slow because it
installs the workspace; the healthcheck allows 120 seconds for that and only
then starts counting failures.

### Configuration

Edit `deploy/.env`. Every value in it is a local default. Two matter before
anything but you can reach the dashboard:

| Variable | Why |
| --- | --- |
| `PASSWORD` | The dashboard login. Its sha256 is what the API compares, so changing it invalidates existing sessions. |
| `POSTGRES_PASSWORD` | Set this if the database is reachable from anywhere but this machine. |

Compose reads that same file twice, which is deliberate: it substitutes
`${VAR}` values like the port mappings, and it is bind-mounted to `/home/.env`
because the application reads its configuration from a *file* in its working
directory rather than from the process environment. Edit it once.

`.env` is gitignored. `deploy/.env.example` is tracked, and every variable the
stack consumes is in it.

### Where your data lives

`workspace/` is mounted at `/home`. That is where your strategies and all
generated data live:

- `workspace/strategies/` — one directory per strategy, each with an
  `__init__.py`. The dashboard's Strategies tab discovers them from here.
- `workspace/storage/` — candles, logs, charts, backtest results.

Everything the app generates is gitignored, so the skeleton stays in version
control and your actual work stays yours. It is a bind mount rather than a
Docker volume on purpose: you can open the directory and see your strategies as
files.

The app refuses to start if its working directory lacks `strategies/` or
`storage/` (`helpers.is_algorithex_project`), which is why `workspace/` ships
with both. If you mount something else at `/home`, it needs those two
directories.

### Deploying the published image

The image is published to GitHub Container Registry under this repository's own
namespace, so the owner needs no credentials:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
```

GHCR packages are private by default. Anyone else pulling needs the package set
to public in the repository's package settings, or a token with `read:packages`.

To run it instead of building from source, set one variable in `deploy/.env`:

```
ALGORITHEX_IMAGE=ghcr.io/<owner>/<repo>:latest
```

### Deploying anywhere else

It is a plain container image with no host requirements beyond Docker. Ports are
`9000` (dashboard), `9001` (LSP), `9002` (MCP) and `8888` (Jupyter). `EXPOSE` is
declared and a `HEALTHCHECK` is baked in, so `docker run -p 9000:9000 ...` and
any orchestrator can tell a serving container from a crash-looping one:

```bash
docker run -p 9000:9000 -v "$PWD/workspace:/home" ghcr.io/<owner>/<repo>:latest
```

Postgres and Redis are needed too. The compose file in `deploy/` is the
supported way to run all three; point it at your own managed databases by
changing `POSTGRES_HOST` and `REDIS_HOST`.

### Running a second instance

The compose project is named explicitly as `algorithex`, and the container names
follow it. Compose otherwise derives the project name from the directory -- so
every checkout calls itself `deploy`, and `docker compose down` in one silently
tears down another.

For a second, independent stack on the same host:

```bash
COMPOSE_PROJECT_NAME=second APP_PORT=9100 docker compose up -d
```

Change the ports as well as the name; two instances cannot share them.

### Continuous integration

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `python-package.yml` | push/PR to `master` | Runs the test suite on Python 3.11 and 3.12 against real Postgres and Redis services. |
| `docker-publish.yml` | push to `master`, release, manual | Publishes `linux/amd64` on every push; `linux/amd64` + `linux/arm64` on a release or manual run. |
| `codeql-analysis.yml` | push/PR to `master`, weekly | Static analysis for Python and JavaScript. |

Nothing in CI publishes to PyPI or Docker Hub. The image goes to GHCR only, into
`ghcr.io/${{ github.repository }}`, so the workflow is correct for whichever
GitHub account it is pushed to without being edited.

The Python matrix is 3.11 and 3.12 because those are the two the suite has
actually been run on: 3.11 is the image's base, 3.12 is what it is run against
interactively. Adding a version means running the suite on it first.

### What this release is and is not

It is a backtester with a read-only live market view. It **cannot place,
modify or cancel orders** — there is no order-placement code path, no request
signing, and no code anywhere that accepts an API key. The live-trading plugin
is not included, so the Live page's session runner is a local simulation stub.

### A note on the frontend

The dashboard ships as a compiled bundle with no source in this repository. The
live market panel inside the Live page is mounted by
`algorithex/static/algorithex-live-feed.js`, which `index.html` loads. If the
frontend bundle is ever rebuilt, `index.html` is regenerated and that script tag
is lost; the panel disappears silently. `tests/test_live_feed_injection.py`
fails when that happens, but the fix is to re-add the one line.

## Screenshots

Captured from this build on `localhost:9000`. Click any image to open it full size.

<table>
<tr>
<td width="50%"><img src="assets/screenshots/01-dashboard-home.png" alt="Dashboard" /><br><sub>Dashboard</sub></td>
<td width="50%"><img src="assets/screenshots/02-live-market.png" alt="Live market data" /><br><sub>Live market data — read-only, Binance &amp; Bybit</sub></td>
</tr>
<tr>
<td><img src="assets/screenshots/03-live-overview.png" alt="Live panel in the dashboard" /><br><sub>Live panel inside the dashboard</sub></td>
<td><img src="assets/screenshots/04-backtest.png" alt="Backtest" /><br><sub>Backtest</sub></td>
</tr>
<tr>
<td><img src="assets/screenshots/05-optimization.png" alt="Optimization" /><br><sub>Optimization</sub></td>
<td><img src="assets/screenshots/06-monte-carlo.png" alt="Monte Carlo" /><br><sub>Monte Carlo</sub></td>
</tr>
<tr>
<td><img src="assets/screenshots/07-strategies.png" alt="Strategies" /><br><sub>Strategies</sub></td>
<td><img src="assets/screenshots/08-validate-workbench.png" alt="Validation workbench" /><br><sub>Validation workbench</sub></td>
</tr>
</table>

## Credits and License

Algorithex is a fork of [Jesse](https://github.com/jesse-ai/jesse) by **Jesse Mir** and contributors, released under the MIT License. Upstream copyright is retained in [`LICENSE`](LICENSE), and the contributors of this fork are credited in [`NOTICE`](NOTICE).

MIT permits use, modification and redistribution provided the original copyright and permission notices are kept — which is why they are still here. If you redistribute this project, keep those files intact.

## Disclaimer
This software is for educational purposes only. USE THE SOFTWARE AT **YOUR OWN RISK**. THE AUTHORS AND ALL AFFILIATES ASSUME **NO RESPONSIBILITY FOR YOUR TRADING RESULTS**. **Do not risk money that you are afraid to lose**. There might be **bugs** in the code - this software DOES NOT come with **ANY warranty**.

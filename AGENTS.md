# Algorithex Repository Guide for AI Agents

## Overview
The algorithex repository is the **core open-source framework** of the Algorithex trading system. It contains the main Python codebase for backtesting trading strategies, importing historical data from crypto exchanges, running optimizations, and providing the API backend for the dashboard. It glues together the other repositories and makes them work together.

## Skills

This repo stores shared **Agent Skills** in `.claude/skills/` — focused, task-specific playbooks used by Claude Code and Codex. Codex also discovers them through the repository-relative `.agents/skills` symlink.

Before starting a task, scan the YAML frontmatter of every `.claude/skills/*/SKILL.md`. When a skill's `description` matches the task, read that `SKILL.md` completely before acting. A matching skill is the authoritative reference for what it covers, and its details are deliberately kept out of this file. This scan is the cross-platform fallback if native skill discovery or symlink support is unavailable.

- **`algorithex-strategy-tests`** — conventions for writing Algorithex strategy and engine tests.

## Key Characteristics

### Central Framework
- **algorithex-live depends on this** - Changes here affect live trading
- **the native indicator kernel integrates here** - Rust functions are called from this codebase
- **dashboard consumes this API** - Frontend uses the FastAPI routes and controller files. 

### Technology Stack
- **Python** - Primary language
- **FastAPI** - API framework for all routes
- **NumPy** - Array operations and calculations
- **keewee** - ORM for the database

## Development Workflow

### Making Changes
When implementing features or fixing bugs:

1. **Understand the scope** - Determine if other repositories such as the dashboard need updates
2. **Implement the code** in the appropriate module
3. **Write/update tests** - Maintain test coverage
4. **Run tests** to verify changes:
   ```bash
   cd algorithex && pytest
   ```
5. **Consider algorithex-live** - Does this affect live trading?
6. **Update API routes** if needed - Follow FastAPI patterns
7. **Don't restart server** unless specifically asked

### Python Environment
Use the Algorithex Python interpreter:
```
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Running Algorithex Backend
The API server provides routes for the dashboard:
```bash
# Stop any running process
pkill -f "algorithex run"

# Start Algorithex from the bot directory
cd my-bot/docker
docker compose up -d algorithex

# Or run the API server directly from the bot directory:
# algorithex run

# Server runs at http://localhost:9001

# Check logs
tail -f /tmp/algorithex-output.log
```

**Important**: Don't restart Algorithex after code changes unless explicitly requested.

### Running Tests
Run the test suite after changes if asked.
```bash
cd algorithex && pytest
```

Dashboard Playwright tests start Algorithex from a disposable project through
`dashboard-v1/tests/e2e/backend.mjs`. That process selects a dedicated env file
with `ALGORITHEX_ENV_FILE`, enables `IS_TEST_ENV=TRUE`, and uses a non-public
`POSTGRES_SCHEMA` containing `e2e`, `test`, or `testing`. Test-only reset/seed
routes are registered only in that mode and still require normal dashboard
authentication. Never enable `RESET_TEST_DATABASE` against `public` or a normal
development schema; the runtime guard intentionally rejects it.

If you've updated the Rust extension, run tests after building:
```bash
cd ../algorithex-rust
./build-local.sh

cd ../algorithex
pytest
```

## Publishing the Docker Image

When the user asks to **"push a docker build for Algorithex"** (or to "release"/"publish"
Algorithex), publish by pushing a version git tag. The build runs on GitHub Actions
(`.github/workflows/docker-publish.yml`) and, on a `v*` tag push, publishes to **PyPI** and
**Docker Hub in parallel**: it uploads the package to PyPI and (independently) builds the
multi-arch `linux/amd64` + `linux/arm64` Docker image, publishing `algorithex/algorithex:<version>`
and `algorithex/algorithex:latest`. The two are independent — if one fails the other still
publishes; just cut a new version to retry the failed half.

Steps (run from inside `algorithex/`):

1. Read the current version from `setup.py` (the `VERSION = "x.y.z"` line). Do **not**
   hardcode it — always read it fresh.
2. Tell the user which version you're about to tag and push (e.g. "Pushing docker build
   for v2.2.0").
3. Confirm the tag doesn't already exist (`git tag -l v<version>` and
   `git ls-remote --tags origin v<version>`). If it already exists, stop and ask the user
   whether to bump the version in `setup.py` (and `version.py`) first.
4. Create and push the tag:
   ```bash
   cd algorithex
   git tag v<version>
   git push origin v<version>
   ```
5. The `push: tags: ['v*']` trigger starts the workflow automatically. Optionally watch it:
   ```bash
   gh run watch --repo algorithex/algorithex
   ```
6. When done, verify both architectures are present:
   ```bash
   docker buildx imagetools inspect algorithex/algorithex:latest
   ```

Notes:
- Pushing a `v*` tag publishes **both PyPI and Docker** via this one workflow — PyPI first,
  Docker second. Do **not** also publish to PyPI manually (e.g. `twine upload`), or the tag
  push will fail on a duplicate-version upload.
- Required `algorithex/algorithex` GitHub repo secrets: `PYPI_API_TOKEN` (PyPI),
  `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN` (Docker Hub), and optionally
  `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (build notification). Nothing to configure locally.
- A manual run (`gh workflow run docker-publish.yml --repo algorithex/algorithex`) **skips PyPI**
  and only rebuilds/pushes `algorithex/algorithex:latest` — useful as a credentials smoke test.

## Important Notes

### Debugging
- **Use `ah.debug()` for all debugging output** - Never use plain `print()`
- **Log format**: `[2024-12-06 18:23:12] ==> Your message here`
- Logs include timestamps and `==>` prefix
- Essential for debugging backtests and live trading sessions

### API Routes
- **Default to POST endpoints** unless specifically asked for GET
- Use FastAPI decorators and patterns
- Follow the structure of existing routes in `algorithex/routes/`
- Return proper HTTP status codes and JSON responses
- Handle errors gracefully

### Code Style
- Add concise comments and docstrings for non-obvious intent, provider quirks, safety boundaries,
  invariants, and tradeoffs. Before handoff, audit every changed file for unexplained thresholds,
  constants, fallback or retry behavior, and test fakes that encode external responses. Keep the
  rationale in the code rather than only in the conversation; do not add comments that merely
  narrate what the code already says.
- Never try to install new packages - assume they're already installed. if need to install new packages, ask me first.
- Follow existing patterns and conventions
- Maintain consistency with the current codebase

### Type Annotations (strategy-facing typing)
User strategies are type-checked in editors (Zed runs Pyrefly), so annotations on anything strategies touch — indicators, `Strategy` properties/attributes, models like `Route` and `Position` — must be precise. Loose framework types surface as false-positive errors in every user's strategy file.

- **Indicators**: any indicator with a `sequential` param that returns `Union[<scalar>, np.ndarray]` must ship this `@overload` trio directly above the implementation (all existing ones already do — keep new ones consistent):
  ```python
  @overload
  def sma(candles: np.ndarray, period: int = ..., source_type: str = ..., sequential: Literal[False] = ...) -> float: ...
  @overload
  def sma(candles: np.ndarray, period: int = ..., source_type: str = ..., sequential: Literal[True] = ...) -> np.ndarray: ...
  @overload
  def sma(candles: np.ndarray, period: int = ..., source_type: str = ..., sequential: bool = ...) -> Union[float, np.ndarray]: ...
  ```
  When changing an indicator's parameters, update all three overloads too — callers only see the overloads, not the implementation signature.
- **Namedtuple-returning indicators** (macd, supertrend, bollinger_bands, …) intentionally keep untyped (Any) fields — don't type their fields piecemeal; it would need a proper generic design to distinguish sequential from non-sequential fields.
- **Late-initialized attributes** that the router/engine sets right after instantiation (`Strategy.name/symbol/exchange/timeframe/position/broker`, `Route.strategy`): declare the real post-initiation type with the `None` default silenced — e.g. `self.position: Position = None  # type: ignore` — instead of `Optional[...]`, so user strategies don't need `is not None` narrowing. `Strategy.hp` is always a `dict` (empty when the strategy has no hyperparameters); check it with falsiness (`if not self.hp`), never `is None`.
- Verify strategy-facing typing changes with Pyrefly:
  ```bash
  cd algorithex
  uvx pyrefly check 'algorithex/indicators/*.py' algorithex/strategies/Strategy.py --search-path .
  ```
  For a broader regression check, run it over `'algorithex/algorithex/strategies/*/__init__.py'` (the test strategies must report 0 errors).
- Try to import only at the top of the file.

### Native Kernel Integration
- Indicator hot paths call a compiled native extension declared in `requirements.txt`; import those symbols directly rather than reimplementing them in Python
- When using native functions, **assume they exist** - don't add existence checks
- Update Python code to call new native implementations
- Run the test suite after changing native call sites to verify integration
- Performance-critical code should be delegated to the native kernel when possible

## File Structure
- `algorithex/` - Main source code
  - `indicators/` - Technical indicators
  - `modes/` - Backtest, optimize, import modes, monte carlo, etc
  - `routes/` - FastAPI route handlers
  - `services/` - data services, etc
  - `strategies/` - Base strategy classes
  - `store/` - State management
- `tests/` - Test suite
- `storage/` - Logs and temporary files
- `requirements.txt` - Python dependencies
- `setup.py` - Package configuration

## Testing Strategy

### Unit Tests
- Run `pytest` after every change if asked in the conversation.
- Maintain or improve test coverage
- Add tests for new features if asked in the conversation.
- Fix failing tests immediately

## Related Repositories
This repository is the foundation of the Algorithex ecosystem:
- **algorithex-rust** - Optional native performance layer for indicators
- **dashboard-v1** - Frontend that consumes Algorithex's API
- **bot** - Algorithex project instance that runs the framework

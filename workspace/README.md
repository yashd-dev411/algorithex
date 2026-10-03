# Workspace

This is the app's working directory. It is mounted at `/home` inside the
container, so anything you put here is visible both on the host and inside the
running app.

It must contain `strategies/` and `storage/`, or the app exits on startup with
*"Current directory is not an Algorithex project."*

- `strategies/` — one directory per strategy, each containing an `__init__.py`
  with your strategy class. The dashboard discovers them from here.
- `storage/` — candles, logs, charts, backtest results. Generated at runtime and
  gitignored, so this stays empty in a fresh clone.

Delete the contents of `strategies/ExampleStrategy/` once you have your own.

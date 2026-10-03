import time
import logging

import click
from importlib.metadata import PackageNotFoundError, version as get_version
import uvicorn

import algorithex.helpers as ah
from algorithex.services.multiprocessing import process_manager
from algorithex.services.web import fastapi_app

# Default Host and Port for the Algorithex API server
HOST = "0.0.0.0"
PORT = 9000


def _get_dist_version() -> str:
    """
    The installed distribution's version, or the source version.

    ``get_version`` raises ``PackageNotFoundError`` when the project is only
    present as a source tree. That is not an exotic state: a fresh ``git
    clone`` that has had ``pip install -r requirements.txt`` run against it has
    no distribution metadata, and this function is called at *import* time (the
    ``@click.version_option`` decorator below), so the failure would take the
    whole package -- and every test module that imports it -- down with it.

    ``algorithex/version.py`` is the single source of the version number and is
    read as text, because importing the package from inside the package would
    recurse straight back into this module.
    """
    try:
        return get_version("algorithex")
    except PackageNotFoundError:
        # Deliberately narrow: a corrupt or unreadable metadata directory is a
        # different problem and should keep propagating rather than be papered
        # over with a number that looks fine.
        from pathlib import Path
        import re

        source = Path(__file__).with_name('version.py').read_text(encoding='utf-8')
        match = re.search(
            r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", source, re.MULTILINE
        )
        return match.group(1) if match else 'unknown'


@click.group()
@click.version_option(_get_dist_version())
def cli() -> None:
    """CLI entrypoint for Algorithex."""
    pass


@cli.command()
@click.option(
    "--strict/--no-strict",
    default=True,
    help="Default is the strict mode which will raise an exception if the values for license is not set.",
)
def install_live(strict: bool) -> None:
    """Install and configure the live trading plugin."""
    from algorithex.services.installer import install

    install(is_live_plugin_already_installed=ah.has_live_trade_plugin(), strict=strict)


def _load_close_prices(path: str):
    """
    Read a close-price column out of a CSV, tolerating the usual header names.

    Kept dependency-free on purpose: a certification report should not need the
    database or the engine to run, only a file.
    """
    import csv

    with open(path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f'{path} is empty')

    header = [cell.strip().lower() for cell in rows[0]]
    index = next(
        (i for i, name in enumerate(header) if name in ('close', 'price', 'adj close')),
        None,
    )
    values = []
    for row in rows[1:] if index is not None else rows:
        cell = row[index] if index is not None and index < len(row) else row[-1]
        try:
            values.append(float(cell))
        except (TypeError, ValueError):
            continue

    if len(values) < 3:
        raise ValueError(
            f'{path} yielded only {len(values)} usable prices; need at least 3'
        )
    import numpy as np

    return np.asarray(values, dtype=np.float64)


@cli.command()
@click.option(
    '--strategy',
    'strategy_path',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Strategy source file to audit.',
)
@click.option(
    '--prices',
    'prices_path',
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help='CSV of close prices. Without it only the static source checks can run.',
)
@click.option(
    '--name',
    'strategy_name',
    default=None,
    help='Name to show in the report. Defaults to the strategy filename.',
)
def certify(strategy_path: str, prices_path, strategy_name) -> None:
    """
    Grade a strategy before you risk money on it.

    Runs the look-ahead audit on the source, and — when a price series is
    supplied — the adversarial robustness and cost-survival checks. Checks with
    no input are reported as NOT RUN and counted against the grade, so a
    partial report never reads like a clean one.
    """
    from algorithex.research.certification import certify as run_certification

    prices = None
    if prices_path:
        try:
            prices = _load_close_prices(prices_path)
        except (OSError, ValueError) as exc:
            raise click.ClickException(f'could not read prices: {exc}')

    report = run_certification(
        strategy_name=strategy_name or strategy_path,
        source_path=strategy_path,
        prices=prices,
    )
    click.echo(report.render())
    if not report.deployable:
        click.echo(
            click.style(
                'Not certified. Checks that were not run are listed above; '
                'supplying that evidence is the next step.',
                fg='yellow',
            )
        )
        raise SystemExit(1)


@cli.command()
@click.option(
    "--skip-agent-rules",
    is_flag=True,
    default=False,
    help="Skip syncing the agent rules file (AGENTS.md / CLAUDE.md / mcp-rules.md) in the project directory.",
)
@click.option(
    "--skip-lsp",
    is_flag=True,
    default=False,
    help="Skip the Python Language Server for this run (disables editor code intelligence).",
)
def run(skip_agent_rules: bool, skip_lsp: bool) -> None:
    """Start the Algorithex application server."""
    from algorithex.services.env import is_test_env

    # Display welcome message
    welcome_message = """
     ██╗███████╗███████╗███████╗███████╗
     ██║██╔════╝██╔════╝██╔════╝██╔════╝
     ██║█████╗  ███████╗███████╗█████╗  
██   ██║██╔══╝  ╚════██║╚════██║██╔══╝  
╚█████╔╝███████╗███████║███████║███████╗
 ╚════╝ ╚══════╝╚══════╝╚══════╝╚══════╝
                                        
    """
    version = _get_dist_version()
    print(welcome_message)

    version_line = (
        click.style("  Algorithex ", fg="white")
        + click.style(f"v{version}", fg="yellow", bold=True)
    )

    if ah.has_live_trade_plugin():
        try:
            from algorithex_live.version import __version__ as live_version
            version_line += (
                click.style("  ·  ", fg="white")
                + click.style("Live Plugin ", fg="white")
                + click.style(f"v{live_version}", fg="yellow", bold=True)
            )
        except ImportError:
            pass

    print(version_line)

    ah.validate_cwd()

    print("")

    # sync the agent rules file (AGENTS.md / CLAUDE.md / mcp-rules.md) with the bundled rules
    if not skip_agent_rules and not is_test_env():
        try:
            from algorithex.mcp.agent_rules import sync_agent_rules

            sync_agent_rules()
            print("")
        except Exception as e:
            print(f"Could not sync agent rules: {str(e)}")
            print("")

    # run all the db migrations
    from algorithex.services.migrator import run as run_migrations
    import peewee

    try:
        run_migrations()
    except peewee.OperationalError:
        sleep_seconds = 10
        print(f"Database wasn't ready. Sleep for {sleep_seconds} seconds and try again.")
        time.sleep(sleep_seconds)
        run_migrations()

    if skip_lsp:
        click.echo("Skipping Python Language Server for this run. Editor code intelligence will be unavailable.")

    if not is_test_env() and not skip_lsp:
        try:
            from algorithex.services.lsp import install_lsp_server

            skip_lsp = not install_lsp_server(allow_skip=True)
        except Exception as e:
            print(ah.color(f"Error installing Python Language Server: {str(e)}", "red"))
            pass

    # read port from .env file and update the global variables port and host, if not found, use default
    global HOST, PORT
    from algorithex.services.env import ENV_VALUES

    if "APP_PORT" in ENV_VALUES:
        PORT = int(ENV_VALUES["APP_PORT"])
    # HOST keeps default value of "0.0.0.0" if not specified

    if "APP_HOST" in ENV_VALUES:
        HOST = ENV_VALUES["APP_HOST"]

    # Set global Algorithex API configuration for MCP and other services

    if not is_test_env() and not skip_lsp:
        try:
            from algorithex.services.lsp import run_lsp_server

            run_lsp_server()
        except Exception as e:
            print(ah.color(f"Error running Python Language Server: {str(e)}", "red"))
            pass
    
    # print dashboard box and suppress uvicorn's own "running on" line
    dashboard_url = f"http://localhost:{PORT}"
    _border = click.style("─" * (len(dashboard_url) + 34), fg="magenta", bold=True)
    print(click.style("┌" + _border + "┐", fg="magenta", bold=True))
    print(
        click.style("│  ", fg="magenta", bold=True)
        + click.style("⬡ Dashboard is available at ", fg="white", bold=True)
        + click.style(dashboard_url, fg="magenta", bold=True)
        + click.style("  │", fg="magenta", bold=True)
    )
    print(click.style("└" + _border + "┘", fg="magenta", bold=True))
    print()

    if not is_test_env():
        try:
            from algorithex.mcp import run_mcp_server

            run_mcp_server(algorithex_host=HOST, algorithex_port=PORT)
        except Exception as e:
            print(ah.color(f"Error running MCP Server: {str(e)}", "red"))
            pass

    # run the main application
    process_manager.flush()

    class _SuppressUvicornStartup(logging.Filter):
        def filter(self, record):
            return "Uvicorn running on" not in record.getMessage()

    logging.getLogger("uvicorn.error").addFilter(_SuppressUvicornStartup())

    uvicorn.run(fastapi_app, host=HOST, port=PORT, log_level="info")

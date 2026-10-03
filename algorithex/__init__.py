import os
import warnings
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from algorithex.services.web import fastapi_app
import algorithex.helpers as ah
from algorithex.services.auth import InvalidAuthError, unauthorized_response

from algorithex.services.e2e_database import reset_test_database_if_requested

reset_test_database_if_requested()

# import cli to register the routes. Do NOT remove this import.
from algorithex.cli import cli


@fastapi_app.exception_handler(InvalidAuthError)
async def invalid_auth_exception_handler(_request, _exc):
    return unauthorized_response()


# to silent stupid pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# get the algorithex directory
ALGORITHEX_DIR = os.path.dirname(os.path.abspath(__file__))

# define lifespan (replaces deprecated @on_event("shutdown"))
@asynccontextmanager
async def lifespan(app):
    yield
    from algorithex.services.db import database
    database.close_connection()
    from algorithex.services.lsp import terminate_lsp_server
    terminate_lsp_server()

fastapi_app.router.lifespan_context = lifespan

class CachedStaticFiles(StaticFiles):
    """
    Static assets with real cache headers.

    Starlette's ``StaticFiles`` sends no ``Cache-Control`` at all, so browsers
    fall back to heuristic revalidation and re-fetch on every navigation. The
    dashboard ships 483 content-hashed JS chunks totalling roughly 24 MB, and
    re-fetching all of that each time is the largest single source of the
    "everything is slow to load" complaint.

    Files under ``/_nuxt/`` carry a content hash in their filename, so a given
    URL can never mean different bytes -- those are safe to cache indefinitely.
    Everything else (``index.html``, ``help.html``) must revalidate, otherwise
    a rebuilt frontend would never reach anyone with the page already open.
    """

    IMMUTABLE_PREFIX = '_nuxt/'
    IMMUTABLE_CACHE = 'public, max-age=31536000, immutable'
    REVALIDATE_CACHE = 'no-cache'

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            # Starlette hands us a path *relative to the mount*, with no leading
            # slash, and built from os.sep -- so it is '_nuxt/a.js' on Linux and
            # '_nuxt\\a.js' on Windows. Normalise before comparing, or the
            # immutable branch silently never matches.
            relative = path.replace('\\', '/').lstrip('/')
            response.headers['Cache-Control'] = (
                self.IMMUTABLE_CACHE
                if relative.startswith(self.IMMUTABLE_PREFIX)
                else self.REVALIDATE_CACHE
            )
        return response


# load homepage
@fastapi_app.get("/")
async def index():
    # Must revalidate: this HTML is what points at the current asset hashes,
    # so caching it is how a rebuilt frontend silently fails to appear.
    response = FileResponse(f"{ALGORITHEX_DIR}/static/index.html")
    response.headers['Cache-Control'] = CachedStaticFiles.REVALIDATE_CACHE
    return response


# Algorithex white-label: local help page so no link leaves the project.
@fastapi_app.get("/help")
async def help_page():
    response = FileResponse(f"{ALGORITHEX_DIR}/static/help.html")
    response.headers['Cache-Control'] = CachedStaticFiles.REVALIDATE_CACHE
    return response


# Algorithex: the strategy validation workbench. Reachable without rebuilding
# the Nuxt bundle, which is why it is served as a standalone page.
@fastapi_app.get("/validate")
async def validate_page():
    response = FileResponse(f"{ALGORITHEX_DIR}/static/validate.html")
    response.headers['Cache-Control'] = CachedStaticFiles.REVALIDATE_CACHE
    return response






# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Routes
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
from algorithex.controllers.websocket_controller import router as websocket_router
from algorithex.controllers.optimization_controller import router as optimization_router
from algorithex.controllers.monte_carlo_controller import router as monte_carlo_router
from algorithex.controllers.exchange_controller import router as exchange_router
from algorithex.controllers.backtest_controller import router as backtest_router
from algorithex.controllers.significance_test_controller import router as significance_test_router
from algorithex.controllers.candles_controller import export_router as candles_export_router, router as candles_router
from algorithex.controllers.strategy_controller import router as strategy_router
from algorithex.controllers.auth_controller import router as auth_router
from algorithex.controllers.config_controller import router as config_router
from algorithex.controllers.notification_controller import router as notification_router
from algorithex.controllers.system_controller import router as system_router
from algorithex.controllers.file_controller import router as file_router
from algorithex.controllers.lsp_controller import router as lsp_router
from algorithex.controllers.closed_trade_controller import router as closed_trade_router
from algorithex.controllers.order_controller import router as order_router
from algorithex.controllers.tabs_controller import router as tabs_router
from algorithex.controllers.period_templates_controller import router as period_templates_router
from algorithex.controllers.route_templates_controller import router as route_templates_router
from algorithex.controllers.ai_model_controller import router as ai_model_router
from algorithex.controllers.data_provider_credentials_controller import router as data_provider_credentials_router
from algorithex.controllers.validation_controller import router as validation_router
from algorithex.services.env import is_test_env

# register routers
fastapi_app.include_router(websocket_router)
fastapi_app.include_router(optimization_router)
fastapi_app.include_router(monte_carlo_router)
fastapi_app.include_router(exchange_router)
fastapi_app.include_router(backtest_router)
fastapi_app.include_router(significance_test_router)
fastapi_app.include_router(candles_router)
fastapi_app.include_router(candles_export_router)
fastapi_app.include_router(strategy_router)
fastapi_app.include_router(auth_router)
fastapi_app.include_router(config_router)
fastapi_app.include_router(notification_router)
fastapi_app.include_router(system_router)
fastapi_app.include_router(file_router)
fastapi_app.include_router(lsp_router)
fastapi_app.include_router(closed_trade_router)
fastapi_app.include_router(order_router)
fastapi_app.include_router(tabs_router)
fastapi_app.include_router(period_templates_router)
fastapi_app.include_router(route_templates_router)
fastapi_app.include_router(ai_model_router)
fastapi_app.include_router(data_provider_credentials_router)
fastapi_app.include_router(validation_router)

if is_test_env():
    from algorithex.controllers.e2e_controller import router as e2e_router
    fastapi_app.include_router(e2e_router)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Live / Paper Trading (Algorithex local mode)
# Always registered so the dashboard never 404s or redirects to a website.
# Real live execution still uses the optional plugin when installed; paper
# sessions run as local simulations without any license token.
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
from algorithex.controllers.live_controller import router as live_router
fastapi_app.include_router(live_router)


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Static Files (Must be loaded at the end to prevent overlapping with API endpoints)
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
fastapi_app.mount("/", CachedStaticFiles(directory=f"{ALGORITHEX_DIR}/static"), name="static")

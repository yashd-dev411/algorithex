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

# load homepage
@fastapi_app.get("/")
async def index():
    return FileResponse(f"{ALGORITHEX_DIR}/static/index.html")


# Algorithex white-label: local help page so no link leaves the project.
@fastapi_app.get("/help")
async def help_page():
    return FileResponse(f"{ALGORITHEX_DIR}/static/help.html")






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
fastapi_app.mount("/", StaticFiles(directory=f"{ALGORITHEX_DIR}/static"), name="static")

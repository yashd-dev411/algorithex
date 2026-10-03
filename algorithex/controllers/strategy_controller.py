from typing import Optional
from fastapi import APIRouter, Header, Query, Depends
from fastapi.responses import JSONResponse
import re

from algorithex.services.auth import require_auth
from algorithex.services.web import (
    NewStrategyRequestJson,
    GetStrategyRequestJson,
    SaveStrategyRequestJson,
    DeleteStrategyRequestJson,
    ForkStrategyRequestJson,
    ImportStrategyRequestJson
)
import algorithex.helpers as ah

router = APIRouter(prefix="/strategy", tags=["Strategy"], dependencies=[Depends(require_auth)])


@router.post("/make")
def make_strategy(json_request: NewStrategyRequestJson) -> JSONResponse:
    """
    Create a new strategy
    """

    from algorithex.services import strategy_handler
    return strategy_handler.generate(json_request.name)


@router.get("/all")
def get_strategies() -> JSONResponse:
    """
    Get all strategies
    """

    from algorithex.services import strategy_handler
    return strategy_handler.get_strategies()


@router.post("/get")
def get_strategy(
        json_request: GetStrategyRequestJson,
) -> JSONResponse:
    """
    Get a specific strategy
    """

    from algorithex.services import strategy_handler
    return strategy_handler.get_strategy(json_request.name)


@router.post("/save")
def save_strategy(
        json_request: SaveStrategyRequestJson,
) -> JSONResponse:
    """
    Save a strategy
    """

    from algorithex.services import strategy_handler
    return strategy_handler.save_strategy(json_request.name, json_request.content)


@router.post("/fork")
def fork_strategy(
        json_request: ForkStrategyRequestJson,
) -> JSONResponse:
    """
    Fork a strategy under a new name
    """

    from algorithex.services import strategy_handler
    return strategy_handler.fork_strategy(json_request.new_name, json_request.content)


@router.post("/delete")
def delete_strategy(
        json_request: DeleteStrategyRequestJson,
) -> JSONResponse:
    """
    Delete a strategy
    """

    from algorithex.services import strategy_handler
    return strategy_handler.delete_strategy(json_request.name)


@router.get("/index")
async def index_algorithex_trade_strategies(
        period: str = Query(...),
        sort_by: str = Query("Sharpe Ratio"),
        submitted_after: Optional[str] = Query(None),
        submitted_before: Optional[str] = Query(None),
        algorithex_trade_token: Optional[str] = Header(None, alias="X-Algorithex-Trade-Token")
) -> JSONResponse:
    """
    Algorithex white-label: browse local strategies instead of a website marketplace
    """
    import os

    strategies_path = os.getcwd() + "/strategies/"
    try:
        names = sorted([
            name for name in os.listdir(strategies_path)
            if os.path.isdir(strategies_path + name) and not name.startswith('.') and name != '__pycache__'
        ])
    except Exception:
        names = []
    data = [{'slug': name, 'name': name, 'mode': 'local'} for name in names]
    return JSONResponse({'status': 'success', 'data': data, 'mode': 'local'})


@router.get("/periods")
async def get_algorithex_trade_periods(
) -> JSONResponse:
    """
    Algorithex white-label: local trading periods
    """
    return JSONResponse({'status': 'success', 'data': ['1m', '5m', '15m', '1h', '4h', '1D'], 'mode': 'local'})


@router.get("/library/{slug}")
async def get_algorithex_trade_strategy(
        slug: str,
        algorithex_trade_token: Optional[str] = Header(None, alias="X-Algorithex-Trade-Token")
) -> JSONResponse:
    """
    Algorithex white-label: get a local strategy by name
    """
    from algorithex.services import strategy_handler
    return strategy_handler.get_strategy(slug)


@router.get("/library/{slug}/metrics")
async def get_algorithex_trade_strategy_metrics(
        slug: str,
        period: str = Query(...),
        symbol: str = Query(...),
        timeframe: str = Query(...),
        algorithex_trade_token: Optional[str] = Header(None, alias="X-Algorithex-Trade-Token")
) -> JSONResponse:
    """
    Algorithex white-label: no hosted metrics; run a local backtest instead.
    """
    return JSONResponse({
        'status': 'success',
        'mode': 'local',
        'message': 'Hosted metrics are disabled in Algorithex local mode. Run a backtest for metrics.',
        'data': {},
    })


@router.post("/import")
async def import_strategy(
        json_request: ImportStrategyRequestJson,
        algorithex_trade_token: Optional[str] = Header(None, alias="X-Algorithex-Trade-Token")
) -> JSONResponse:
    """
    Algorithex white-label: import from a local strategy name (copy it).
    """
    from algorithex.services import strategy_handler

    res = strategy_handler.get_strategy(json_request.slug)
    if res.status_code != 200:
        return JSONResponse({
            'status': 'error',
            'message': f'Local strategy "{json_request.slug}" not found. Place code under strategies/{json_request.slug}/__init__.py.'
        }, status_code=404)
    return JSONResponse({'status': 'success', 'message': 'Local strategy already available', 'mode': 'local'})

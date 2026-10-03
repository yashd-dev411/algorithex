from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import psutil
import os

from algorithex.services.web import FeedbackRequestJson, ReportExceptionRequestJson, HelpSearchRequestJson
from algorithex.services.multiprocessing import process_manager
from algorithex.services import algorithex_trade
from algorithex.services.general_info import get_external_general_info, get_general_info, get_local_general_info
from algorithex.services.auth import require_auth
from algorithex.version import __version__ as algorithex_version
from algorithex.models import BacktestSession, OptimizationSession, LiveSession, MonteCarloSession, SignificanceTestSession
import algorithex.helpers as ah

router = APIRouter(prefix="/system", tags=["System"], dependencies=[Depends(require_auth)])


@router.post("/feedback")
def feedback(json_request: FeedbackRequestJson) -> JSONResponse:
    """
    Save feedback locally for the Algorithex team
    """

    return algorithex_trade.feedback(json_request.description, json_request.email)


@router.post("/report-exception")
def report_exception(json_request: ReportExceptionRequestJson) -> JSONResponse:
    """
    Save an exception report locally for the Algorithex team
    """

    return algorithex_trade.report_exception(
        json_request.description,
        json_request.traceback,
        json_request.mode,
        json_request.attach_logs,
        json_request.session_id,
        json_request.email,
        has_live=ah.has_live_trade_plugin()
    )


@router.post("/general-info")
def general_info() -> JSONResponse:
    """
    Get general information about the system
    """

    try:
        data = get_general_info(has_live=ah.has_live_trade_plugin())
    except Exception as e:
        ah.error(str(e))
        return JSONResponse({
            'error': str(e)
        }, status_code=500)

    return JSONResponse(
        data,
        status_code=200
    )


@router.post("/general-info/local")
def local_general_info() -> JSONResponse:
    try:
        data = get_local_general_info(has_live=ah.has_live_trade_plugin())
    except Exception as e:
        ah.error(str(e))
        return JSONResponse({
            'error': str(e)
        }, status_code=500)

    return JSONResponse(data, status_code=200)


@router.post("/general-info/external")
def external_general_info() -> JSONResponse:
    try:
        data = get_external_general_info(has_live=ah.has_live_trade_plugin())
    except Exception as e:
        ah.error(str(e))
        return JSONResponse({
            'error': str(e)
        }, status_code=500)

    return JSONResponse(data, status_code=200)


@router.post("/active-workers")
def active_workers() -> JSONResponse:
    """
    Get a list of active workers
    """

    return JSONResponse({
        'data': list(process_manager.active_workers)
    }, status_code=200)


@router.post("/help-search")
def help_search(json_request: HelpSearchRequestJson) -> JSONResponse:
    """
    Algorithex white-label: local help search (no website dependency).
    """
    query = (json_request.query or '').strip()
    return JSONResponse({
        'results': [],
        'query': query,
        'mode': 'local',
        'message': 'Online help search is disabled in Algorithex local mode. See local docs and strategy examples.',
    }, status_code=200)


@router.get("/system-info")
def system_info() -> JSONResponse:
    """
    Get system info (CPU, RAM, version, etc)
    """

    strategies_path = os.getcwd() + "/strategies/"
    try:
        strategies = list(sorted([name for name in os.listdir(strategies_path) if os.path.isdir(strategies_path + name) and not name.startswith('.')]))
        if "__pycache__" in strategies:
            strategies.remove("__pycache__")
        strategy_count = len(strategies)
    except Exception:
        strategy_count = 0

    return JSONResponse({
        'cpu_cores': psutil.cpu_count(logical=True),
        'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
        'ram_usage_percent': psutil.virtual_memory().percent,
        'ram_total_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
        'ram_used_gb': round(psutil.virtual_memory().used / (1024 ** 3), 2),
        'algorithex_version': algorithex_version,
        'strategy_count': strategy_count
    }, status_code=200)


@router.get("/user-activity")
def user_activity() -> JSONResponse:
    """
    Get user activity stats (backtests, optimizations, etc)
    """

    current_timestamp = ah.now_to_timestamp(True)
    day_ago = current_timestamp - (24 * 60 * 60 * 1000)
    week_ago = current_timestamp - (7 * 24 * 60 * 60 * 1000)

    def get_counts(model):
        try:
            all_time = model.select().where(model.status != 'draft').count()
            last_24h = model.select().where((model.status != 'draft') & (model.created_at >= day_ago)).count()
            last_7d = model.select().where((model.status != 'draft') & (model.created_at >= week_ago)).count()
            return {'all_time': all_time, 'last_24h': last_24h, 'last_7d': last_7d}
        except Exception:
            return {'all_time': 0, 'last_24h': 0, 'last_7d': 0}

    try:
        backtests = get_counts(BacktestSession)
        optimizations = get_counts(OptimizationSession)
        live_sessions = get_counts(LiveSession)
        monte_carlo = get_counts(MonteCarloSession)
        significance_tests = get_counts(SignificanceTestSession)
    except Exception as e:
        ah.error(str(e))
        backtests = optimizations = live_sessions = monte_carlo = significance_tests = {'all_time': 0, 'last_24h': 0, 'last_7d': 0}

    return JSONResponse({
        'backtests': backtests,
        'optimizations': optimizations,
        'live_sessions': live_sessions,
        'monte_carlo': monte_carlo,
        'significance_tests': significance_tests
    }, status_code=200)

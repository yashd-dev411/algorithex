import requests
from fastapi.responses import JSONResponse
from algorithex.services.auth import get_access_token
import algorithex.helpers as ah
import json
from algorithex.info import ALGORITHEX_API_URL


def feedback(description: str, email: str = None) -> JSONResponse:
    # Algorithex white-label: store feedback locally instead of posting to a website.
    try:
        import os
        from datetime import datetime, timezone
        os.makedirs('storage/logs', exist_ok=True)
        with open('storage/logs/feedback.log', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} email={email} {description}\n")
    except Exception as e:
        return JSONResponse({
            'status': 'error',
            'message': f"Could not save feedback locally: {str(e)}"
        }, status_code=200)

    return JSONResponse({
        'status': 'success',
        'message': 'Feedback saved locally'
    }, status_code=200)


def report_exception(
        description: str, traceback: str, mode: str, attach_logs: bool, session_id: str, email: str = None, has_live: bool = False
) -> JSONResponse:
    access_token = get_access_token()

    if attach_logs and session_id:
        path_exchange_log = None
        if mode == 'backtest':
            path_log = f'storage/logs/backtest-mode/{session_id}.txt'
        elif mode == 'live':
            path_log = f'storage/logs/live-mode/{session_id}.txt'
            path_exchange_log = f'storage/logs/live-mode/{session_id}-raw-exchange-logs.txt'
        elif mode == 'significance-test':
            path_log = f'storage/logs/significance-test-mode/{session_id}.txt'
        else:
            raise ValueError('Invalid mode')

        # attach exchange_log if there's any
        files = {}
        if ah.file_exists(path_log):
            files['log_file'] = open(path_log, 'rb')
        if path_exchange_log and ah.file_exists(path_exchange_log):
            files['exchange_log'] = open(path_exchange_log, 'rb')
            
        if not files:
            files = None
    else:
        files = None

    from algorithex.version import __version__ as algorithex_version
    info = {
        'os': ah.get_os(),
        'python_version': '{}.{}'.format(*ah.python_version()),
        'is_docker': ah.is_docker(),
        'algorithex_version': algorithex_version
    }
    if has_live:
        from algorithex_live.version import __version__ as live_plugin_version
        info['live_plugin_version'] = live_plugin_version

    # Algorithex white-label: store exception reports locally.
    try:
        import os
        from datetime import datetime, timezone
        os.makedirs('storage/logs', exist_ok=True)
        with open('storage/logs/exceptions.log', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} mode={mode} email={email} info={json.dumps(info)} desc={description} traceback={traceback}\n")
    except Exception as e:
        return JSONResponse({
            'status': 'error',
            'message': f"Could not save exception report locally: {str(e)}"
        }, status_code=200)

    return JSONResponse({
        'status': 'success',
        'message': 'Exception report saved locally'
    }, status_code=200)

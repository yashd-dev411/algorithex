import hmac
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256

import requests
import algorithex.helpers as jh
from algorithex.info import exchange_info, algorithex_supported_timeframes, ALGORITHEX_API_URL, ALGORITHEX_API2_URL
from algorithex.services.env import ENV_VALUES, is_dev_env


UPDATE_INFO_CACHE_SECONDS = 15 * 60


def _get_plan_info(access_token: str | None) -> tuple[dict, dict]:
    # Algorithex white-label: everything runs locally with premium limits,
    # no website account or license token required.
    local_limits = {
        'ip_limit': -1,
        'live_trading_tabs': -1,
        'trading_routes': -1,
        'data_routes': -1,
        'timeframes': -1,
        'exchanges': {},
    }
    if not access_token:
        return {'plan': 'premium'}, local_limits

    try:
        response = requests.post(
            ALGORITHEX_API_URL + '/v2/user-info',
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=10
        )

        content_type = response.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            raise Exception(
                f"Algorithex API returned unexpected content type '{content_type}'. "
                f"The service might be temporarily unavailable. Please try again later."
            )

        if response.status_code != 200:
            try:
                error_message = response.json().get('message', 'Unknown error')
            except ValueError:
                error_message = f"Received status code {response.status_code}"
            raise Exception(
                f"Failed to get user info from Algorithex API: {error_message}"
            )

        plan_info = response.json()
        return plan_info, plan_info['limits']
    except requests.exceptions.RequestException as e:
        jh.debug(f"Algorithex local mode: plan backend unreachable, using premium ({str(e)})")
        return {'plan': 'premium'}, {
            'ip_limit': -1,
            'live_trading_tabs': -1,
            'trading_routes': -1,
            'data_routes': -1,
            'timeframes': -1,
            'exchanges': {},
        }
    except ValueError as e:
        jh.debug(f"Algorithex local mode: invalid plan response, using premium ({str(e)})")
        return {'plan': 'premium'}, {
            'ip_limit': -1,
            'live_trading_tabs': -1,
            'trading_routes': -1,
            'data_routes': -1,
            'timeframes': -1,
            'exchanges': {},
        }


@lru_cache(maxsize=2)
def _get_update_info(_cache_bucket: int) -> dict:
    update_info = {}

    try:
        # Algorithex white-label: no external update checks, report local version.
        from algorithex.version import __version__ as local_version
        update_info['algorithex_latest_version'] = local_version
        update_info['is_update_info_available'] = True
        return update_info

        response = requests.get('https://pypi.org/pypi/algorithex/json', timeout=10)
        if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
            update_info['algorithex_latest_version'] = response.json()['info']['version']
        else:
            raise ValueError("Invalid response from PyPI")

        response = requests.get(
            ALGORITHEX_API2_URL + '/plugins/live/releases/info',
            timeout=10
        )
        if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
            update_info['algorithex_live_latest_version'] = response.json()[0]['version']
        else:
            raise ValueError("Invalid response from Algorithex API")

        update_info['is_update_info_available'] = True
    except Exception as e:
        update_info['is_update_info_available'] = False
        jh.debug(f"Failed to fetch update info: {str(e)}")

    return update_info


def _get_bootstrap_context_id(access_token: str | None, algorithex_version: str, live_version: str) -> str:
    payload = '\0'.join((access_token or 'guest', algorithex_version, live_version))
    return hmac.new(
        ENV_VALUES['PASSWORD'].encode('utf-8'),
        payload.encode('utf-8'),
        sha256
    ).hexdigest()


def _get_live_plugin_info(has_live: bool, access_token: str | None) -> tuple[bool, str]:
    plugin_installed = False
    live_version = ''
    if has_live:
        try:
            from algorithex_live.version import __version__ as live_version
            plugin_installed = True
        except ImportError:
            plugin_installed = False

    return plugin_installed, live_version


def _format_limits(limits: dict) -> dict:
    if not limits:
        return {}

    return {
        'ip_limit': limits.get('ip_limit'),
        'live_trading_tabs': limits.get('live_trading_tabs'),
        'trading_routes': limits.get('trading_routes'),
        'data_routes': limits.get('data_routes'),
        'timeframes': limits.get('timeframes'),
        'exchanges': list(limits.get('exchanges', {}).keys()),
    }


def get_local_general_info(has_live=False) -> dict:
    from algorithex.services.auth import get_access_token
    from algorithex.version import __version__ as algorithex_version

    access_token = get_access_token()
    plugin_installed, live_version = _get_live_plugin_info(has_live, access_token)

    system_info = {
        'algorithex_version': algorithex_version,
        'python_version': '{}.{}'.format(*jh.python_version()),
        'operating_system': jh.get_os(),
        'cpu_cores': jh.cpu_cores_count(),
        'is_docker': jh.is_docker(),
    }
    if access_token and live_version:
        system_info['live_plugin_version'] = live_version

    strategies_path = os.getcwd() + "/strategies/"
    strategies = list(sorted([
        name for name in os.listdir(strategies_path)
        if os.path.isdir(strategies_path + name) and not name.startswith('.')
    ]))
    if "__pycache__" in strategies:
        strategies.remove("__pycache__")

    # Algorithex white-label: no license token exists locally, yet every
    # feature is unlocked. Report premium + licensed so the dashboard never
    # shows website auth gates or upgrade redirects.
    return {
        'exchanges': exchange_info,
        'strategies': strategies,
        'algorithex_supported_timeframes': algorithex_supported_timeframes,
        'has_live_plugin_installed': plugin_installed,
        'has_license_token': True,
        'plan': 'premium',
        'limits': {
            'ip_limit': -1,
            'live_trading_tabs': -1,
            'trading_routes': -1,
            'data_routes': -1,
            'timeframes': -1,
            'exchanges': [],
        },
        'system_info': system_info,
        'bootstrap_context_id': _get_bootstrap_context_id(access_token, algorithex_version, live_version),
    }


def get_external_general_info(has_live=False) -> dict:
    from algorithex.services.auth import get_access_token
    from algorithex.version import __version__ as algorithex_version

    access_token = get_access_token()
    _, live_version = _get_live_plugin_info(has_live, access_token)

    cache_bucket = int(time.monotonic() // UPDATE_INFO_CACHE_SECONDS)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            plan_future = executor.submit(_get_plan_info, access_token)
            update_future = executor.submit(_get_update_info, cache_bucket)
            plan_info, limits = plan_future.result()
            update_info = update_future.result().copy()
    except Exception as e:
        # Algorithex white-label: never fail the dashboard when offline.
        jh.debug(f"Algorithex local mode: external info unavailable, using premium ({str(e)})")
        plan_info, limits = {'plan': 'premium'}, {}
        update_info = {'is_update_info_available': False}

    res = {
        'update_info': update_info,
        'plan': plan_info['plan'],
        'bootstrap_context_id': _get_bootstrap_context_id(access_token, algorithex_version, live_version),
    }

    if limits:
        res['limits'] = _format_limits(limits)

    return res


def get_general_info(has_live=False) -> dict:
    return {
        **get_local_general_info(has_live),
        **get_external_general_info(has_live),
    }

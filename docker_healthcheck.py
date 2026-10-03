"""
Container healthcheck.

Exits 0 when the dashboard is serving, 1 otherwise. Used by the image's
HEALTHCHECK instruction and by the compose healthcheck, so `docker ps` and any
orchestrator can tell a running service from one that is crash-looping.

Why a file rather than a one-liner in the Dockerfile
---------------------------------------------------
The obvious `python -c "urlopen(...)"` prints a full traceback on every failed
probe. A container that is still booting produces one of those every 30 seconds
for a minute and a half, which buries the actual error in the logs and makes a
slow start look like a crash. This prints one line and exits.

`/` is the probe because it needs no authentication and is served by the
application itself. It is a better signal than a bare port check: the app binds
late, after the workspace is installed and the database and Redis are reachable,
so a successful response means the service is genuinely ready rather than merely
listening.

The port is read from APP_PORT rather than hardcoded, because the deploy template
invites changing it to avoid colliding with something already on the host.
"""

import os
import sys
import urllib.error
import urllib.request

DEFAULT_PORT = 9000
ENV_FILE = '/home/.env'
TIMEOUT = 4


def _port_from_env_file() -> str | None:
    """Read APP_PORT out of the project's .env file.

    The app resolves its listen port from a .env file in its working directory,
    so that file is the authority. The process environment usually carries it
    too -- compose injects it -- but the file is checked as well because a
    bare `docker run` does not inject anything.
    """
    try:
        with open(ENV_FILE, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                if key.strip() == 'APP_PORT' and value.strip():
                    return value.strip()
    except OSError:
        return None
    return None


def resolve_port() -> int:
    """The port the app is actually listening on.

    This must not be hardcoded. The deploy template invites changing APP_PORT
    so the dashboard does not collide with something else on the host, and a
    healthcheck pointed at the default would report a container serving on 9100
    as permanently dead -- which reads as a broken deployment rather than a
    mismatched probe.
    """
    raw = os.environ.get('APP_PORT') or _port_from_env_file()
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


def main() -> int:
    url = f'http://127.0.0.1:{resolve_port()}/'
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            if response.status == 200:
                return 0
            print(f'healthcheck: {url} returned HTTP {response.status}', file=sys.stderr)
            return 1
    except urllib.error.HTTPError as exc:
        print(f'healthcheck: {url} returned HTTP {exc.code}', file=sys.stderr)
        return 1
    except Exception as exc:
        # One line. The type and message are what matter; the stack trace of a
        # refused connection is noise repeated on every probe.
        print(f'healthcheck: {url} unreachable ({type(exc).__name__}: {exc})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

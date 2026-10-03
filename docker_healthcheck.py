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
"""

import sys
import urllib.error
import urllib.request

URL = 'http://127.0.0.1:9000/'
TIMEOUT = 4


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT) as response:
            if response.status == 200:
                return 0
            print(f'healthcheck: {URL} returned HTTP {response.status}', file=sys.stderr)
            return 1
    except urllib.error.HTTPError as exc:
        print(f'healthcheck: {URL} returned HTTP {exc.code}', file=sys.stderr)
        return 1
    except Exception as exc:
        # One line. The type and message are what matter; the stack trace of a
        # refused connection is noise repeated on every probe.
        print(f'healthcheck: {URL} unreachable ({type(exc).__name__}: {exc})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

import algorithex.helpers as jh
from algorithex.services import logger as algorithex_logger
import threading
import traceback
import os
from algorithex.services.redis import sync_publish
from algorithex.repositories import live_session_repository
from algorithex.store import store
from algorithex.enums import live_session_statuses


def _terminal_debug(message: str) -> None:
    try:
        jh.terminal_debug(message)
    except Exception:
        pass


def register_custom_exception_handler() -> None:
    # other threads
    def handle_thread_exception(args) -> None:
        if args.exc_type == SystemExit:
            return

        formatted_traceback = ''.join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )

        if args.exc_type.__name__ == 'Termination':
            sync_publish('termination', {})
            jh.terminate_app()
        else:
            # send notifications if it's a live session
            if jh.is_live():
                try:
                    algorithex_logger.error(
                        f'{args.exc_type.__name__}: {args.exc_value}'
                    )
                    algorithex_logger.info(formatted_traceback)
                except Exception as e:
                    _terminal_debug(
                        f'Error logging uncaught thread exception: {type(e).__name__}: {e}\n{formatted_traceback}'
                    )

                # Store exception in live session
                try:
                    live_session_repository.store_live_session_exception(
                        store.app.session_id,
                        f"{args.exc_type.__name__}: {str(args.exc_value)}",
                        formatted_traceback
                    )
                    live_session_repository.update_live_session_status(store.app.session_id, live_session_statuses.STOPPED)
                    live_session_repository.update_live_session_finished(store.app.session_id)
                except Exception as e:
                    _terminal_debug(f'Error storing live session exception: {type(e).__name__}: {e}')

            try:
                sync_publish('exception', {
                    'error': f"{args.exc_type.__name__}: {str(args.exc_value)}",
                    'traceback': formatted_traceback
                })
            finally:
                terminate_session()

    threading.excepthook = handle_thread_exception


def terminate_session(error: str = '', traceback_str: str = ''):
    try:
        sync_publish('unexpectedTermination', {
            'message': "Session terminated as the result of an uncaught exception",
        })
    except Exception as e:
        _terminal_debug(f'Error publishing unexpected session termination: {type(e).__name__}: {e}')

    try:
        algorithex_logger.error('Session terminated as the result of an uncaught exception')
    except Exception as e:
        _terminal_debug(f'Error logging unexpected session termination: {type(e).__name__}: {e}')

    # Algorithex robustness: a crashed backtest worker must not leave the
    # dashboard spinning on "running" forever. Resolve this worker's own
    # session id and mark a still-running backtest as failed with the error.
    try:
        from algorithex.services.multiprocessing import process_manager
        client_id = process_manager.get_client_id(os.getpid())
        if client_id:
            from algorithex.models.BacktestSession import (
                get_backtest_session_by_id,
                store_backtest_session_exception,
                update_backtest_session_status,
            )
            try:
                session = get_backtest_session_by_id(client_id)
            except Exception:
                session = None
            if session is not None and getattr(session, 'status', '') == 'running':
                msg = error or 'Terminated: uncaught exception in worker (see server logs)'
                store_backtest_session_exception(client_id, msg, traceback_str or '')
                update_backtest_session_status(client_id, 'failed')
    except Exception as e:
        _terminal_debug(f'Error marking crashed backtest session as failed: {type(e).__name__}: {e}')

    try:
        jh.terminate_app()
    except BaseException as e:
        _terminal_debug(f'Error closing resources during session termination: {type(e).__name__}: {e}')
    finally:
        os._exit(1)

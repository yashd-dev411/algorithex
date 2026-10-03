from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from algorithex.services.env import ENV_VALUES
from algorithex.services import auth as authenticator
from algorithex.services.auth import require_auth
from algorithex.services.multiprocessing import process_manager
from algorithex.services.web import LoginRequestJson
import algorithex.helpers as ah

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login")
def login(json_request: LoginRequestJson):
    """
    Authenticate user with password and return a token
    """
    return authenticator.password_to_token(json_request.password)


@router.post("/user-validation")
def login(json_request: LoginRequestJson):
    """
    Authenticate user with password and return a token
    """
    return authenticator.user_validation(json_request.password)


@router.post("")
def auth(json_request: LoginRequestJson):
    """
    Authenticate user with password and return a token
    """
    return authenticator.password_to_token(json_request.password)


@router.post("/shutdown", dependencies=[Depends(require_auth)])
async def shutdown(background_tasks: BackgroundTasks):
    """
    Shutdown the application
    """

    background_tasks.add_task(ah.terminate_app)
    return JSONResponse({'message': 'Shutting down...'})


@router.post("/local-token", dependencies=[Depends(require_auth)])
async def algorithex_trade_token():
    """
    Algorithex white-label: no website account needed. Return a local token
    so the dashboard unlocks premium features offline.
    """
    from hashlib import sha256

    license_token = ENV_VALUES.get('LICENSE_API_TOKEN') or 'local'
    access_token = sha256(f"algorithex-local:{license_token}".encode('utf-8')).hexdigest()

    return JSONResponse({
        'status': 'success',
        'access_token': access_token,
        'user': {'plan': 'premium', 'mode': 'local'}
    })

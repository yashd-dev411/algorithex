from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocket, WebSocketDisconnect
from algorithex.services import auth as authenticator
from algorithex.services.ws_manager import ws_manager
import algorithex.helpers as ah

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    from algorithex.services.env import ENV_VALUES

    if not authenticator.is_valid_token(token):
        return

    # Use connection manager to handle this websocket
    connection_id = str(id(websocket))
    ah.terminal_debug(f"WebSocket {connection_id} connecting")
    
    await ws_manager.connect(websocket)
    channel_pattern = f"{ENV_VALUES['APP_PORT']}:channel:*"
    
    # Start Redis listener if not already started
    await ws_manager.start_redis_listener(channel_pattern)
    
    try:
        # Keep the connection alive and handle pong responses
        while True:
            message = await websocket.receive_text()
            try:
                data = ah.json_loads(message)
                # Handle pong responses for heartbeat
                if data.get('type') == 'pong':
                    pass
            except:
                pass
    except WebSocketDisconnect:
        ah.terminal_debug(f"WebSocket {connection_id} disconnected")
        ws_manager.disconnect(websocket)
        # Optionally stop Redis listener if no more clients
        await ws_manager.stop_redis_listener()
    except Exception as e:
        ah.terminal_debug(f"WebSocket error: {str(e)}")
        ws_manager.disconnect(websocket)
        await ws_manager.stop_redis_listener()

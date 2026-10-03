from algorithex import exceptions
import algorithex.helpers as ah
from algorithex.services import logger


def validate_routes(router) -> None:
    if not router.routes:
        raise exceptions.InvalidRoutes(
            'No routes found. Please add at least one route at: routes.py')

    # validation for number of routes in the live mode
    if ah.is_live():
        if len(router.routes) > 10:
            logger.broadcast_error_without_logging('Too many routes (not critical, but use at your own risk): Using that more than 5 routes in live/paper trading is not recommended because exchange WS connections are often not reliable for handling that much traffic.')

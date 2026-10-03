import algorithex.helpers as ah
import algorithex.services.logger as logger
from algorithex.store import store


def set_up():
    store.reset()

# TODO
# def test_can_log_error_by_firing_event():
#     set_up()
#
#     # fire first error event
#     logger.error('first error!!!!!')
#     first_logged_error = {'id': 0, 'time': ah.now_to_timestamp(), 'message': 'first error!!!!!'}
#
#     assert store.logs.errors == [first_logged_error]
#
#     # fire second error event
#     logger.error('second error!!!!!')
#     second_logged_error = {'id': 1, 'time': ah.now_to_timestamp(), 'message': 'second error!!!!!'}
#
#     assert store.logs.errors == [first_logged_error, second_logged_error]
#
# TODO
# def test_can_log_info_by_firing_event():
#     set_up()
#
#     # fire first info event
#     logger.info('first info!!!!!')
#     first_logged_info = {'id': 0, 'time': ah.now_to_timestamp(), 'message': 'first info!!!!!'}
#
#     assert store.logs.info == [first_logged_info]
#
#     # fire second info event
#     logger.info('second info!!!!!')
#     second_logged_info = {
#         'id': 1,
#         'time': ah.now_to_timestamp(),
#         'message': 'second info!!!!!'
#     }
#
#     assert store.logs.info == [first_logged_info, second_logged_info]

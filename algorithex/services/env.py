from dotenv import load_dotenv, dotenv_values
import algorithex.helpers as ah
import os
import sys

# fix directory issue
sys.path.insert(0, os.getcwd())

ENV_VALUES = {}

if ah.is_unit_testing():
    ENV_VALUES['POSTGRES_HOST'] = '127.0.0.1'
    ENV_VALUES['POSTGRES_NAME'] = 'algorithex_db'
    ENV_VALUES['POSTGRES_PORT'] = '5432'
    ENV_VALUES['POSTGRES_USERNAME'] = 'algorithex_user'
    ENV_VALUES['POSTGRES_PASSWORD'] = 'password'
    ENV_VALUES['REDIS_HOST'] = 'localhost'
    ENV_VALUES['REDIS_PORT'] = '6379'
    ENV_VALUES['REDIS_DB'] = 0
    ENV_VALUES['REDIS_PASSWORD'] = ''
    ENV_VALUES['APP_PORT'] = 3000
    ENV_VALUES['IS_DEV_ENV'] = 'TRUE'
    ENV_VALUES['LSP_PORT'] = 9001
    ENV_VALUES['MCP_PORT'] = 9002

if ah.is_algorithex_project():
    env_path = os.environ.get('ALGORITHEX_ENV_FILE', '.env')
    load_dotenv(dotenv_path=env_path)

    # create and expose ENV_VALUES
    ENV_VALUES = dotenv_values(env_path)

    # validation for existence of the selected env file
    if len(list(ENV_VALUES.keys())) == 0:
        # NOTE: print directly here instead of ah.error(). This code runs at import time,
        # before the rest of Algorithex is initialized, and env.py is imported by
        # algorithex.services.redis. ah.error() calls is_live() -> algorithex.config ->
        # algorithex.modes.utils -> algorithex.services.logger -> algorithex.services.redis, which is
        # still mid-import — so ah.error() here triggers a circular ImportError that hides
        # this very message. A plain stderr print + os._exit keeps the message visible.
        print(
            f'{env_path} is missing or empty. '
            'This usually happens when you\'re in the wrong directory. '
            '\n\nIf you haven\'t created an Algorithex project yet, do that by running: \n'
            'algorithex make-project {name}\n'
            'And then go into that project, and run the same command.',
            file=sys.stderr,
        )
        os._exit(1)

    if not ah.is_unit_testing() and ENV_VALUES['PASSWORD'] == '':
        raise EnvironmentError('You forgot to set the PASSWORD in your .env file')


def is_dev_env() -> bool:
    return ENV_VALUES.get('IS_DEV_ENV', '').upper() == 'TRUE'


def is_test_env() -> bool:
    return ENV_VALUES.get('IS_TEST_ENV', '').upper() == 'TRUE'

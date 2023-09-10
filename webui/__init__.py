from functools import lru_cache
from config import Config
from i18n import I18nAuto


MENU_ITEMS = {
    "Get help": "https://github.com/SayanoAI/RVC-Studio/discussions",
    "Report a Bug": "https://github.com/SayanoAI/RVC-Studio/issues",
    "About": """This project provides a comprehensive platform for training RVC models and generating AI voice covers.
    Check out this github for more info: https://github.com/SayanoAI/RVC-Studio
    """
}

@lru_cache
def load_config():
    return Config(), I18nAuto()

config, i18n = load_config()
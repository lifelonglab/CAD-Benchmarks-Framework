import logging
import pathlib

from cadbench.paths import OUTPUT_PATH


def setup_logger(level: int = logging.INFO, logs_path: pathlib.Path = OUTPUT_PATH / 'logs.log'):
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=level,
                        handlers=[logging.FileHandler(logs_path, mode='w'), logging.StreamHandler()], force=True)
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up with level: %s", logging.getLevelName(level))

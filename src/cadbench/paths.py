import os
import pathlib
import logging
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()

load_dotenv()

RESOURCES_PATH = pathlib.Path(
    os.environ["RESOURCES_PATH"]) if "RESOURCES_PATH" in os.environ else PROJECT_ROOT / "resources"
logger.info(f"Using resources path: {RESOURCES_PATH}")

TABULAR_DATASETS_PATH = RESOURCES_PATH / 'datasets' / 'tabular'
TIMESERIES_DATASETS_PATH = RESOURCES_PATH / 'datasets' / 'timeseries'

OUTPUT_PATH = pathlib.Path(
    os.environ["OUTPUT_PATH"]) if "OUTPUT_PATH" in os.environ else PROJECT_ROOT / "output"
OUTPUT_DATASETS_PATH = OUTPUT_PATH / 'datasets'



def create_path(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

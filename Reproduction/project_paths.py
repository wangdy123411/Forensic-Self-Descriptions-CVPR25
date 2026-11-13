import os
from pathlib import Path
import sys


EXTENSION_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXTENSION_ROOT.parent
PROJECT_ROOT = Path(os.environ.get("FSD_PROJECT_ROOT", EXTENSION_ROOT)).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("FSD_DATA_ROOT", PROJECT_ROOT / "Data")).expanduser().resolve()
ARCHIVE_ROOT = Path(os.environ.get("FSD_ARCHIVE_ROOT", DATA_ROOT)).expanduser().resolve()
WEIGHTS_ROOT = Path(os.environ.get("FSD_WEIGHTS_DIR", PROJECT_ROOT / "Weights/fsd_v1.2.0")).expanduser().resolve()
INPUTS_ROOT = Path(os.environ.get("FSD_INPUTS_ROOT", PROJECT_ROOT / "inputs")).expanduser().resolve()
REFERENCE_PATH = Path(os.environ.get("FSD_PAPER_REFERENCE", EXTENSION_ROOT / "configs/paper_reference.json")).expanduser().resolve()

for directory in (CODE_ROOT, EXTENSION_ROOT, EXTENSION_ROOT / "zeroshot_validation", EXTENSION_ROOT / "reproduce"):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

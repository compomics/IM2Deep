from pathlib import Path
import logging

from deeplc import DeepLC
from tensorflow.keras.models import load_model


LOGGER = logging.getLogger(__name__)

def predict(psm_file, output_file=None, model_name="tims", n_jobs=None):
    """Predict CCS values from a PSM file."""
    if model_name == "tims":
        path_model = Path(__file__).parent / "models" / "TIMS"
        models = [
            load_model(path_model / "model_1.h5"),
        ]  # TODO: load all models in this folder
    else:
        LOGGER.error(f"Model type {model_name} not available.")

    dlc = DeepLC(path_model=path_model, n_jobs=n_jobs, predict_ccs=True)


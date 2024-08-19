from pathlib import Path
import logging

import torch
import pandas as pd
import numpy as np

from im2deeptrainer.model import IM2DeepMultiTransfer
from im2deeptrainer.extract_data import _get_matrices
from im2deeptrainer.utils import FlexibleLossSorted
from im2deep.utils import multi_config

LOGGER = logging.getLogger(__name__)
MULTI_CKPT_PATH = Path(__file__).parent / "models" / "TIMS_multi" / "TIMS_multi.ckpt"
REFERENCE_DATASET_PATH = Path(__file__).parent / "reference_data" / "multi_reference.gz"


def get_ccs_shift_multi(
    df_cal: pd.DataFrame, reference_dataset: pd.DataFrame, use_charge_state: int = 2
) -> float:
    """Calculate CCS shift factor for multi predictions.

    Parameters
    ----------
    df_cal
        Peptides with CCS values.
    reference_dataset
        Reference dataset with CCS values.
    use_charge_state
        Charge state to use for CCS shift calculation, by default 2.

    Returns CCS shift factor
    """
    LOGGER.debug(
        f"Using charge state {use_charge_state} for calibration of multiconformer predictions."
    )

    reference_tmp = reference_dataset[reference_dataset["charge"] == use_charge_state]
    df_tmp = df_cal[df_cal["charge"] == use_charge_state]

    both = pd.merge(
        left=reference_tmp,
        right=df_tmp,
        on=["seq", "modifications"],
        how="inner",
        suffixes=("_ref", "_data"),
    )

    LOGGER.debug(
        f"Calculating CCS shift based on {both.shape[0]} overlapping peptide-charge pairs between PSMs and reference dataset."
    )

    return 0 if both.empty else np.mean(both["ccs_observed"] - both["CCS"])


def get_ccs_shift_per_charge_multi(
    df_cal: pd.DataFrame, reference_dataset: pd.DataFrame
) -> np.ndarray:
    """Calculate CCS shift factor per charge state for multiconformational predictions

    Parameters
    ----------
    df_cal
        Peptides with CCS values.
    reference_dataset
        Reference dataset with CCS values.

    Returns
    -------
    np.ndarray
        CCS shift factor per charge state.
    """
    both = pd.merge(
        left=reference_dataset,
        right=df_cal,
        on=["seq", "modifications", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )
    return both.groupby("charge").apply(lambda x: np.mean(x["ccs_observed"] - x["CCS"])).to_dict()


def calculate_ccs_shift_multi(
    df_cal: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge: bool = True,
    use_charge_state: int = None,
) -> float:
    """
    Apply CCS shift to CCS values for multiconformational predictions.

    Parameters
    ----------
    df_cal
        Peptides with CCS values.
    reference_dataset
        Reference dataset with CCS values.
    per_charge
        Apply CCS shift per charge state, by default True.
    use_charge_state
        Charge state to use for CCS shift calculation, needs to be [2,4], by default None.

    Returns
    -------
    float
        CCS shift factor.
    """
    df_cal = df_cal[(df_cal["charge"] < 5) & (df_cal["charge"] > 1)]

    if not per_charge:
        shift_factor = get_ccs_shift_multi(df_cal, reference_dataset, use_charge_state)
        LOGGER.debug(f"CCS shift factor: {shift_factor}")
        return shift_factor

    else:
        shift_factor_dict = get_ccs_shift_per_charge_multi(df_cal, reference_dataset)
        LOGGER.debug(f"CCS shift factors: {shift_factor_dict}")
        return shift_factor_dict


def linear_calibration_multi(
    df_pred: pd.DataFrame,
    df_cal: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge: bool = True,
    use_charge_state: int = None,
) -> pd.DataFrame:
    """
    Calibrate multiconformer predictions using linear calibration.

    Parameters
    ----------
    df_pred
        Peptides with CCS predictions.
    df_cal
        Peptides with CCS values.
    reference_dataset
        Reference dataset with CCS values.
    per_charge
        Apply calibration per charge state, by default True.
    use_charge_state
        Charge state to use for calibration, needs to be [2,4], by default None.

    Returns
    -------
    pd.DataFrame
        Calibrated PSMs.
    """
    LOGGER.info("Calibrating multiconformer predictions using linear calibration...")
    if per_charge:
        LOGGER.info("Generating general shift factor for multiconformer predictions...")
        general_shift = calculate_ccs_shift_multi(
            df_cal, reference_dataset, per_charge=False, use_charge_state=use_charge_state
        )
        LOGGER.info("Getting shift factors per charge state...")
        shift_factor_dict = calculate_ccs_shift_multi(df_cal, reference_dataset, per_charge=True)

        df_pred["shift_multi"] = df_pred["charge"].map(shift_factor_dict).fillna(general_shift)
        df_pred["predicted_ccs_multi_1"] = (
            df_pred["predicted_ccs_multi_1"] + df_pred["shift_multi"]
        )
        df_pred["predicted_ccs_multi_2"] = (
            df_pred["predicted_ccs_multi_2"] + df_pred["shift_multi"]
        )

    else:
        shift_factor = calculate_ccs_shift_multi(
            df_cal, reference_dataset, per_charge=False, use_charge_state=use_charge_state
        )
        df_pred["predicted_ccs_multi_1"] = df_pred["predicted_ccs_multi_1"] + shift_factor
        df_pred["predicted_ccs_multi_2"] = df_pred["predicted_ccs_multi_2"] + shift_factor

    LOGGER.info("Multiconformer predictions calibrated.")
    return df_pred


def predict_multi(df_pred, df_cal, output_file, calibrate_per_charge, use_charge_state):
    criterion = FlexibleLossSorted()
    model = IM2DeepMultiTransfer.load_from_checkpoint(
        MULTI_CKPT_PATH, config=multi_config, criterion=criterion
    )

    df_pred["tr"] = 0  # Placeholder for DeepLC compatibility
    matrices = _get_matrices(df_pred)

    tensors = {}
    for key in matrices:
        tensors[key] = torch.tensor(matrices[key]).type(torch.FloatTensor)

    dataset = torch.utils.data.TensorDataset(*[tensors[key] for key in tensors])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=multi_config["batch_size"], shuffle=False
    )

    model.eval()
    with torch.no_grad():
        preds = []
        for index, batch in enumerate(dataloader):
            prediction = model.predict_step(batch)
            preds.append(prediction)
        predictions = torch.cat(preds).numpy()

    df_pred["predicted_ccs_multi_1"] = predictions[:, 0]
    df_pred["predicted_ccs_multi_2"] = predictions[:, 1]

    if df_cal is not None:
        df_pred = linear_calibration_multi(
            df_pred,
            df_cal,
            reference_dataset=pd.read_csv(
                REFERENCE_DATASET_PATH, compression="gzip", keep_default_na=False
            ),
            per_charge=calibrate_per_charge,
            use_charge_state=use_charge_state,
        )

    return df_pred[["predicted_ccs_multi_1", "predicted_ccs_multi_2"]]

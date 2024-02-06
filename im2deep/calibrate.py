import logging

import numpy as np
import pandas as pd
from numpy import ndarray
from psm_utils.peptidoform import Peptidoform

LOGGER = logging.getLogger(__name__)

def im2ccs(reverse_im, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    Convert ion mobility to CCS.  #TODO: Took this from ionmob. how to reference?

    Parameters
    ----------
    reverse_im : float
        Reduced ion mobility.
    mz : float
        Precursor m/z.
    charge : int
        Precursor charge.
    mass_gas : float, optional
        Mass of gas, by default 28.013
    temp : float, optional
        Temperature in Celsius, by default 31.85
    t_diff : float, optional
        Factor to convert Celsius to Kelvin, by default 273.15"""

    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) * 1 / reverse_im)


def get_ccs_shift(
    cal_df: pd.DataFrame, reference_dataset: pd.DataFrame, use_charge_state: int = 2
) -> float:
    """
    Calculate CCS shift factor, i.e. a constant offset,
    based on identical precursors as in reference dataset.

    Parameters
    ----------
    cal_df : pd.DataFrame
        PSMs with CCS values.
    reference_dataset : pd.DataFrame
        Reference dataset with CCS values.
    use_charge_state : int, optional
        Charge state to use for CCS shift calculation, needs to be [2,4], by default 2.
    return_shift_factor : float
        CCS shift factor.

    """
    LOGGER.debug(f"Using charge state {use_charge_state} for CCS shift calculation.")

    tmp_df = cal_df.copy(deep=True)
    tmp_ref_df = reference_dataset.copy(deep=True)

    tmp_df["sequence"] = tmp_df["peptidoform"].apply(lambda x: x.proforma.split("\\")[0])
    tmp_df["charge"] = tmp_df["peptidoform"].apply(lambda x: x.precursor_charge)
    tmp_ref_df["sequence"] = tmp_ref_df["peptidoform"].apply(
        lambda x: Peptidoform(x).proforma.split("\\")[0]
    )
    tmp_ref_df["charge"] = tmp_ref_df["peptidoform"].apply(
        lambda x: Peptidoform(x).precursor_charge
    )

    reference_tmp = tmp_ref_df[tmp_ref_df["charge"] == use_charge_state]
    df_tmp = tmp_df[tmp_df["charge"] == use_charge_state]
    both = pd.merge(
        left=reference_tmp,
        right=df_tmp,
        right_on=["sequence", "charge"],
        left_on=["sequence", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )
    LOGGER.debug(
        """Calculating CCS shift based on {} overlapping peptide-charge pairs
        between PSMs and reference dataset""".format(both.shape[0])
    )
    LOGGER.debug(both.columns)
    # How much CCS in calibration data is larger than reference CCS, so predictions
    # need to be increased by this amount
    return 0 if both.shape[0] == 0 else np.mean(both["ccs_observed"] - both["CCS"])


def get_ccs_shift_per_charge(cal_df: pd.DataFrame, reference_dataset: pd.DataFrame) -> ndarray:
    """
    Calculate CCS shift factor per charge state,
    i.e. a constant offset based on identical precursors as in reference.

    Parameters
    ----------
    df : pd.DataFrame
        PSMs with CCS values.
    reference_dataset : pd.DataFrame
        Reference dataset with CCS values.

    Returns
    -------
    ndarray
        CCS shift factors per charge state.

    """
    tmp_df = cal_df.copy(deep=True)
    tmp_ref_df = reference_dataset.copy(deep=True)

    tmp_df["sequence"] = tmp_df["peptidoform"].apply(lambda x: x.proforma.split("\\")[0])
    tmp_df["charge"] = tmp_df["peptidoform"].apply(lambda x: x.precursor_charge)
    tmp_ref_df["sequence"] = tmp_ref_df["peptidoform"].apply(
        lambda x: Peptidoform(x).proforma.split("\\")[0]
    )
    tmp_ref_df["charge"] = tmp_ref_df["peptidoform"].apply(
        lambda x: Peptidoform(x).precursor_charge
    )

    reference_tmp = tmp_ref_df
    df_tmp = tmp_df
    both = pd.merge(
        left=reference_tmp,
        right=df_tmp,
        right_on=["sequence", "charge"],
        left_on=["sequence", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )
    return both.groupby("charge").apply(lambda x: np.mean(x["ccs_observed"] - x["CCS"])).to_dict()


def calculate_ccs_shift(
    cal_df: pd.DataFrame, reference_dataset: pd.DataFrame, per_charge=True, use_charge_state=None
) -> float:
    """
    Apply CCS shift to CCS values.

    Parameters
    ----------
    psm_list : PSMList

    """
    cal_df['charge'] = cal_df['peptidoform'].apply(lambda x: x.precursor_charge)
    cal_df = cal_df[cal_df["charge"] < 5]  # predictions do not go higher for IM2Deep
    # Filter df for high_conf_hits TODO: Check if possible, however this is necessary for MS2Rescore
    # cal_df = cal_df[cal_df["spectrum_id"].isin(high_conf_hits)]

    if not per_charge:
        shift_factor = get_ccs_shift(
            cal_df,
            reference_dataset,
            use_charge_state=use_charge_state,
        )
        LOGGER.debug(f"CCS shift factor: {shift_factor}")
        return shift_factor

    else:
        shift_factor_dict = get_ccs_shift_per_charge(cal_df, reference_dataset)
        LOGGER.debug(f"CCS shift factor dict: {shift_factor_dict}")
        return shift_factor_dict


def linear_calibration(
    preds_df: pd.DataFrame,
    calibration_dataset: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge=True,
    use_charge_state=None,
) -> pd.DataFrame:
    """
    Calibrate PSM df using linear calibration.

    Parameters
    ----------
    psm_df : pd.DataFrame


    """
    LOGGER.info("Calibrating CCS values using linear calibration...")
    preds_df["charge"] = preds_df["peptidoform"].apply(lambda x: x.precursor_charge)
    if per_charge:
        general_shift = calculate_ccs_shift(
            calibration_dataset, reference_dataset, per_charge=False, use_charge_state=use_charge_state
        )
        shift_factor_dict = calculate_ccs_shift(
            calibration_dataset, reference_dataset, per_charge=True
        )
        for charge in preds_df["charge"].unique():
            if charge not in shift_factor_dict:
                LOGGER.info("No overlapping precursors for charge state {}. Using overall shift factor for precursors with that charge.".format(charge))
                shift_factor_dict[charge] = general_shift
        LOGGER.info("Shift factors per charge: {}".format(shift_factor_dict))
        preds_df["predicted_ccs_calibrated"] = preds_df.apply(
            lambda x: x["predicted_ccs"] + shift_factor_dict[x["charge"]], axis=1
        )
    else:
        shift_factor = calculate_ccs_shift(
            calibration_dataset, reference_dataset, per_charge=False, use_charge_state=use_charge_state
        )
        preds_df["predicted_ccs_calibrated"] = preds_df.apply(
            lambda x: x["predicted_ccs"] + shift_factor, axis=1
        )
    LOGGER.info("CCS values calibrated.")
    return preds_df

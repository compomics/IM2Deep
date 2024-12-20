import logging
from pathlib import Path

import pandas as pd
from deeplc import DeepLC
from psm_utils.psm_list import PSMList

from im2deep.calibrate import linear_calibration
from im2deep.utils import ccs2im

LOGGER = logging.getLogger(__name__)
REFERENCE_DATASET_PATH = Path(__file__).parent / "reference_data" / "reference_ccs.zip"


# TODO: get file reading out of the function
def predict_ccs(
    psm_list_pred: PSMList,
    psm_list_cal_df=None,
    file_reference=REFERENCE_DATASET_PATH,
    output_file=None,
    model_name="tims",
    calibrate_per_charge=True,
    use_charge_state=2,
    use_single_model=True,
    n_jobs=None,
    write_output=True,
    ion_mobility=False,
):
    """Run IM2Deep."""
    LOGGER.info("IM2Deep started.")
    reference_dataset = pd.read_csv(file_reference)

    if model_name == "tims":
        path_model = Path(__file__).parent / "models" / "TIMS"

    path_model_list = list(path_model.glob("*.keras"))
    if use_single_model:
        LOGGER.debug("Using model {}".format(path_model_list[2]))
        path_model_list = [path_model_list[2]]

    dlc = DeepLC(path_model=path_model_list, n_jobs=n_jobs, predict_ccs=True)
    LOGGER.info("Predicting CCS values...")
    preds = dlc.make_preds(psm_list=psm_list_pred, calibrate=False)
    LOGGER.info("CCS values predicted.")
    psm_list_pred_df = psm_list_pred.to_dataframe()
    psm_list_pred_df["predicted_ccs"] = preds
    psm_list_pred_df["charge"] = psm_list_pred_df["peptidoform"].apply(
        lambda x: x.precursor_charge
    )

    if psm_list_cal_df is not None:
        psm_list_pred_df = linear_calibration(
            psm_list_pred_df,
            calibration_dataset=psm_list_cal_df,
            reference_dataset=reference_dataset,
            per_charge=calibrate_per_charge,
            use_charge_state=use_charge_state,
        )

    if ion_mobility:
        LOGGER.info("Converting CCS to IM values...")
        psm_list_pred_df["predicted_im"] = ccs2im(
            psm_list_pred_df["predicted_ccs"],
            psm_list_pred_df["peptidoform"].apply(lambda x: x.theoretical_mz),
            psm_list_pred_df["charge"],
        )

    if write_output:
        if not ion_mobility:
            LOGGER.info("Writing output file...")
            output_file = open(output_file, "w")
            output_file.write("modified_seq,charge,predicted CCS\n")
            for peptidoform, charge, CCS in zip(
                psm_list_pred_df["peptidoform"],
                psm_list_pred_df["charge"],
                psm_list_pred_df["predicted_ccs"],
            ):
                output_file.write(f"{peptidoform},{charge},{CCS}\n")
            output_file.close()
        else:
            LOGGER.info("Writing output file...")
            output_file = open(output_file, "w")
            output_file.write("modified_seq,charge,predicted IM\n")
            for peptidoform, charge, IM in zip(
                psm_list_pred_df["peptidoform"],
                psm_list_pred_df["charge"],
                psm_list_pred_df["predicted_im"],
            ):
                output_file.write(f"{peptidoform},{charge},{IM}\n")
            output_file.close()

    LOGGER.info("IM2Deep finished!")

    if not ion_mobility:
        return psm_list_pred_df["predicted_ccs"]
    else:
        return psm_list_pred_df["predicted_im"]

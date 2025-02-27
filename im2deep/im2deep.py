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
    multi=False,
    calibrate_per_charge=True,
    use_charge_state=2,
    use_single_model=True,
    n_jobs=None,
    write_output=True,
    ion_mobility=False,
    pred_df=None,
    cal_df=None,
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

    if multi:
        from im2deep.predict_multi import predict_multi

        LOGGER.info("Predicting multiconformer CCS values...")
        pred_df = predict_multi(
            psm_list_pred,
            cal_df,
            calibrate_per_charge,
            use_charge_state,
        )

    if ion_mobility:
        if not multi:
            psm_list_pred_df["predicted_im"] = ccs2im(
                psm_list_pred_df["predicted_ccs"],
                psm_list_pred_df["peptidoform"].apply(lambda x: x.theoretical_mz),
                psm_list_pred_df["charge"],
            )
            if write_output:
                LOGGER.info("Writing output file for ion mobility prediction...")
                with open(output_file, "w") as f:
                    f.write("modified_seq,charge,predicted IM\n")
                    for peptidoform, charge, IM in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_im"],
                    ):
                        f.write(f"{peptidoform},{charge},{IM}\n")
            LOGGER.info("IM2Deep finished!")
            return psm_list_pred_df["predicted_im"]
        else:
            psm_list_pred_df["predicted_im"] = ccs2im(
                psm_list_pred_df["predicted_ccs"],
                psm_list_pred_df["peptidoform"].apply(lambda x: x.theoretical_mz),
                psm_list_pred_df["charge"],
            )
            psm_list_pred_df["predicted_im_multi_1"] = ccs2im(
                pred_df["predicted_ccs_multi_1"],
                psm_list_pred_df["peptidoform"].apply(lambda x: x.theoretical_mz),
                psm_list_pred_df["charge"],
            )
            psm_list_pred_df["predicted_im_multi_2"] = ccs2im(
                pred_df["predicted_ccs_multi_2"],
                psm_list_pred_df["peptidoform"].apply(lambda x: x.theoretical_mz),
                psm_list_pred_df["charge"],
            )
            if write_output:
                LOGGER.info("Writing output file for multi-ion mobility prediction...")
                with open(output_file, "w") as f:
                    f.write(
                        "modified_seq,charge,predicted IM single,predicted IM multi 1,predicted IM multi 2\n"
                    )
                    for peptidoform, charge, IM_single, IM_multi_1, IM_multi_2 in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_im"],
                        psm_list_pred_df["predicted_im_multi_1"],
                        psm_list_pred_df["predicted_im_multi_2"],
                    ):
                        f.write(f"{peptidoform},{charge},{IM_single},{IM_multi_1},{IM_multi_2}\n")
            LOGGER.info("IM2Deep finished!")
            return psm_list_pred_df["predicted_im"]
    else:
        if not multi:
            if write_output:
                LOGGER.info("Writing output file for CCS prediction...")
                with open(output_file, "w") as f:
                    f.write("modified_seq,charge,predicted CCS\n")
                    for peptidoform, charge, CCS in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_ccs"],
                    ):
                        f.write(f"{peptidoform},{charge},{CCS}\n")
            LOGGER.info("IM2Deep finished!")
            return psm_list_pred_df["predicted_ccs"]
        else:
            if write_output:
                LOGGER.info("Writing output file for multi-CCS prediction...")
                with open(output_file, "w") as f:
                    f.write(
                        "modified_seq,charge,predicted CCS single,predicted CCS multi 1,predicted CCS multi 2\n"
                    )
                    for peptidoform, charge, CCS_single, CCS_multi_1, CCS_multi_2 in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_ccs"],
                        pred_df["predicted_ccs_multi_1"],
                        pred_df["predicted_ccs_multi_2"],
                    ):
                        f.write(
                            f"{peptidoform},{charge},{CCS_single},{CCS_multi_1},{CCS_multi_2}\n"
                        )
            LOGGER.info("IM2Deep finished!")
            return psm_list_pred_df["predicted_ccs"]

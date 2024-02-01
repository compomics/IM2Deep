"""Command line interface to IM2Deep."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from deeplc import DeepLC
from psm_utils.io import read_file
from psm_utils.io.exceptions import PSMUtilsIOException
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from rich.logging import RichHandler

from im2deep._exceptions import IM2DeepError
from im2deep.calibrate import linear_calibration

REFERENCE_DATASET_PATH = Path(__file__).parent / "reference_data" / "reference_ccs.zip"

LOGGER = logging.getLogger(__name__)


def setup_logging(passed_level):
    log_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if passed_level.lower() not in log_mapping:
        raise ValueError(
            f"""Invalid log level: {passed_level}.
                         Should be one of {log_mapping.keys()}"""
        )

    logging.basicConfig(
        level=log_mapping[passed_level.lower()],
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

#Command line arguments TODO: Make config_parser script
@click.command()
@click.argument("psm_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-c",
    "--calibration_file",
    type=click.Path(exists=False),
    default=None,
    help="Calibration file name.",
)
@click.option(
    "-o",
    "--output_file",
    type=click.Path(exists=False),
    default=None,
    help="Output file name.",
)
@click.option(
    "-m",
    "--model_name",
    type=click.Choice(["tims"]),
    default="tims",
    help="Model name.",
)
@click.option(
    "-l",
    "--log_level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    help="Logging level.",
)
@click.option(
    "-n",
    "--n_jobs",
    type=click.INT,
    default=None,
    help="Number of jobs to use for parallel processing.",
)
@click.option(
    "--calibrate_per_charge",
    type=click.BOOL,
    default=True,
    help="Calibrate CCS values per charge state.",
)
@click.option(
    "--use_charge_state",
    type=click.INT,
    default=2,
    help="Charge state to use for calibration. Only used if calibrate_per_charge is set to False.",
)

def main(
    psm_file: str,
    calibration_file: Optional[str] = None,
    output_file: Optional[str] = None,
    model_name: Optional[str] = "tims",
    log_level: Optional[str] = "info",
    n_jobs: Optional[int] = None,
    calibrate_per_charge: Optional[bool] = True,
    use_charge_state: Optional[int] = 2,
):
    """Command line interface to IM2Deep."""
    setup_logging(log_level)
    try:
        run(psm_file, calibration_file, output_file, model_name, calibrate_per_charge, use_charge_state, n_jobs)
    except IM2DeepError as e:
        LOGGER.error(e)
        sys.exit(1)


def run(
    file_pred,
    file_cal=None,
    file_pred_out=None,
    model_name="tims",
    calibrate_per_charge=True,
    use_charge_state=2,
    n_jobs=None,
):
    """Run IM2Deep."""
    LOGGER.info("IM2Deep started.")
    reference_dataset = pd.read_csv(REFERENCE_DATASET_PATH)

    with open(file_pred) as f:
        first_line_pred = f.readline().strip()
    if file_cal:
        with open(file_cal) as fc:
            first_line_cal = fc.readline().strip()

    if "modifications" in first_line_pred.split(",") and "seq" in first_line_pred.split(","):
        # Read input file
        df_pred = pd.read_csv(file_pred)
        df_pred.fillna("", inplace=True)

        list_of_psms = []
        for seq, mod, charge, ident in zip(
            df_pred["seq"], df_pred["modifications"], df_pred["charge"], df_pred.index
        ):
            list_of_psms.append(
                PSM(peptidoform=peprec_to_proforma(seq, mod, charge), spectrum_id=ident)
            )
        psm_list_pred = PSMList(psm_list=list_of_psms)

    else:
        # psm_list_pred = read_file(file_pred)
        try:
            psm_list_pred = read_file(file_pred)
        except PSMUtilsIOException:
            LOGGER.error("Invalid input file. Please check the format of the input file.")
            sys.exit(1)

    psm_list_cal = []
    if (
        file_cal
        and "modifications" in first_line_cal.split(",")
        and "seq" in first_line_cal.split(",")
    ):
        df_cal = pd.read_csv(file_cal)
        df_cal.fillna("", inplace=True)
        del file_cal

        list_of_cal_psms = []
        for seq, mod, charge, ident, CCS in zip(
            df_cal["seq"], df_cal["modifications"], df_cal["charge"], df_cal.index, df_cal["CCS"]
        ):
            list_of_cal_psms.append(
                PSM(peptidoform=peprec_to_proforma(seq, mod, charge), spectrum_id=ident)
            )
        psm_list_cal = PSMList(psm_list=list_of_cal_psms)
        psm_list_cal_df = psm_list_cal.to_dataframe()
        psm_list_cal_df["observed_ccs"] = df_cal["CCS"]
        del df_cal

    else:
        LOGGER.error("Invalid calibration file. Please check the format of the calibration file.")
        sys.exit(1)

    if model_name == "tims":
        path_model = Path(__file__).parent / "models" / "TIMS"

    path_model_list = list(path_model.glob("*.hdf5"))

    dlc = DeepLC(path_model=path_model_list, n_jobs=n_jobs, predict_ccs=True)
    LOGGER.info("Predicting CCS values...")
    preds = dlc.make_preds(psm_list=psm_list_pred, calibrate=False)
    LOGGER.info("CCS values predicted.")
    psm_list_pred_df = psm_list_pred.to_dataframe()
    psm_list_pred_df["predicted_ccs"] = preds

    calibrated_psm_list_pred_df = linear_calibration(
        psm_list_pred_df,
        calibration_dataset=psm_list_cal_df,
        reference_dataset=reference_dataset,
        per_charge=calibrate_per_charge,
        use_charge_state=use_charge_state,
    )

    LOGGER.info("Writing output file...")
    if file_pred_out:
        file_pred_out = open(file_pred_out, "w")
        file_pred_out.write("seq,modifications,charge,predicted CCS\n")
        for seq, mod, charge, ident, CCS in zip(
            df_pred["seq"],
            df_pred["modifications"],
            df_pred["charge"],
            df_pred.index,
            calibrated_psm_list_pred_df["predicted_ccs_calibrated"],
        ):
            file_pred_out.write(f"{seq},{mod},{charge},{CCS}\n")
        file_pred_out.close()
    else:
        #Get path of psm file
        output_file = Path(file_pred).parent / (Path(file_pred).stem + "_IM2Deep-predictions.csv")
        LOGGER.info("Writing output file to %s", output_file)
        output_file = open(output_file, "w")
        output_file.write("seq,modifications,charge,predicted CCS\n")
        for seq, mod, charge, ident, CCS in zip(
            df_pred["seq"],
            df_pred["modifications"],
            df_pred["charge"],
            df_pred.index,
            calibrated_psm_list_pred_df["predicted_ccs_calibrated"],
        ):
            output_file.write(f"{seq},{mod},{charge},{CCS}\n")
        output_file.close()

    LOGGER.info("IM2Deep finished!")


if __name__ == "__main__":
    main()

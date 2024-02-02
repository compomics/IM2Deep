"""Command line interface to IM2Deep."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd

# from deeplc import DeepLC
from psm_utils.io import read_file
from psm_utils.io.exceptions import PSMUtilsIOException
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from rich.logging import RichHandler

from im2deep._exceptions import IM2DeepError
from im2deep.im2deep import predict_ccs

# from im2deep.calibrate import linear_calibration

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


# Command line arguments TODO: Make config_parser script
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

    with open(psm_file) as f:
        first_line_pred = f.readline().strip()
    if calibration_file:
        with open(calibration_file) as fc:
            first_line_cal = fc.readline().strip()

    if "modifications" in first_line_pred.split(",") and "seq" in first_line_pred.split(","):
        # Read input file
        df_pred = pd.read_csv(psm_file)
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
            psm_list_pred = read_file(psm_file)
        except PSMUtilsIOException:
            LOGGER.error("Invalid input file. Please check the format of the input file.")
            sys.exit(1)

    psm_list_cal = []
    if (
        calibration_file
        and "modifications" in first_line_cal.split(",")
        and "seq" in first_line_cal.split(",")
    ):
        df_cal = pd.read_csv(calibration_file)
        df_cal.fillna("", inplace=True)
        del calibration_file

        list_of_cal_psms = []
        for seq, mod, charge, ident, CCS in zip(
            df_cal["seq"],
            df_cal["modifications"],
            df_cal["charge"],
            df_cal.index,
            df_cal["CCS"],
        ):
            list_of_cal_psms.append(
                PSM(peptidoform=peprec_to_proforma(seq, mod, charge), spectrum_id=ident)
            )
        psm_list_cal = PSMList(psm_list=list_of_cal_psms)
        psm_list_cal_df = psm_list_cal.to_dataframe()
        psm_list_cal_df["ccs_observed"] = df_cal["CCS"]
        del df_cal

    else:
        LOGGER.error(
            "Invalid calibration file. Please check the format of the calibration file."
        )
        sys.exit(1)

    if not output_file:
        output_file = Path(psm_file).parent / (Path(psm_file).stem + "_IM2Deep-predictions.csv")
    try:
        predict_ccs(
            psm_list_pred,
            psm_list_cal_df,
            output_file=output_file,
            model_name=model_name,
            calibrate_per_charge=calibrate_per_charge,
            use_charge_state=use_charge_state,
            n_jobs=n_jobs,
        )
    except IM2DeepError as e:
        LOGGER.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

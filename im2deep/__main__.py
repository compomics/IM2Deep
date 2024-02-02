"""Command line interface to IM2Deep."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
# import pandas as pd
# from deeplc import DeepLC
# from psm_utils.io import read_file
# from psm_utils.io.exceptions import PSMUtilsIOException
# from psm_utils.io.peptide_record import peprec_to_proforma
# from psm_utils.psm import PSM
# from psm_utils.psm_list import PSMList
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
        predict_ccs(psm_file, calibration_file, REFERENCE_DATASET_PATH, output_file, model_name, calibrate_per_charge, use_charge_state, n_jobs)
    except IM2DeepError as e:
        LOGGER.error(e)
        sys.exit(1)

if __name__ == "__main__":
    main()

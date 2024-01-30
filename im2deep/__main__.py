"""Command line interface to IM2Deep."""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import logging
import sys

from rich.logging import RichHandler
import click
import pandas as pd
import numpy as np
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from psm_utils.io import read_file
from psm_utils.io import write_file
from deeplc import DeepLC

from im2deep._exceptions import IM2DeepError
from im2deep.im2deep import predict
from im2deep.calibrate import linear_calibration

# TODO: Change to a compressed format? Ask ralf
REFERENCE_DATASET = pd.read_csv(
    Path(__file__).parent / "reference_data" / "peprec_ccs_with_index.csv"
)
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


@click.command()
@click.argument("psm_file", type=click.Path(exists=True, dir_okay=False))
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
def main(
    psm_file: str,
    output_file: Optional[str] = None,
    model_name: Optional[str] = "tims",
    log_level: Optional[str] = "info",
):
    """Command line interface to IM2Deep."""
    # psm_file = Path(psm_file)
    # if not output_file:
    #     output_file = psm_file.parent / Path(psm_file.stem + "-im2deep-predictions.csv")
    # predictions = predict(psm_file, model_name)
    # predictions.to_csv(output_file, index=False)
    setup_logging(log_level)
    LOGGER.info(Path(__file__).parent.joinpath("models"))
    try:
        run(psm_file, output_file, model_name)
    except IM2DeepError as e:
        LOGGER.error(e)
        sys.exit(1)


def run(
    file_pred,
    file_pred_out=None,
    model_name="tims",
    calibrate_per_charge=True,
    use_charge_state=2,
    n_jobs=None,
):
    """Run IM2Deep."""
    with open(file_pred) as f:
        first_line_pred = f.readline().strip()

    if "modifications" in first_line_pred.split(",") and "seq" in first_line_pred.split(","):
        # Read input file
        df_pred = pd.read_csv(file_pred)
        df_pred.fillna("", inplace=True)
        del file_pred

        list_of_psms = []
        for seq, mod, charge, ident in zip(
            df_pred["seq"], df_pred["modifications"], df_pred["charge"], df_pred.index
        ):
            list_of_psms.append(
                PSM(peptidoform=peprec_to_proforma(seq, mod, charge), spectrum_id=ident)
            )
        psm_list_pred = PSMList(psm_list=list_of_psms)

    else:
        psm_list_pred = read_file(file_pred)

    if model_name == "tims":
        path_model = Path(__file__).parent / "models" / "TIMS"

    dlc = DeepLC(path_model=path_model, n_jobs=None, predict_ccs=True)
    preds = dlc.make_preds(seq_df=psm_list_pred, calibrate=False)
    psm_list_pred_df = psm_list_pred.to_dataframe()
    psm_list_pred_df["predicted_ccs"] = preds

    calibrated_psm_list_pred = linear_calibration(
        psm_list_pred_df, reference_dataset=REFERENCE_DATASET, per_charge=calibrate_per_charge
    )


if __name__ == "__main__":
    main()

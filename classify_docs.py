import click

from classifier_settings import (
    BATCH_SIZE, NUM_WORKERS, CLASSES,
    PREDICT_DEVICE, RESULT_COLUMNS
)
from run_classifier import classify


@click.command()
@click.argument("path",
                type=click.Path(resolve_path=True, dir_okay=False),
                required=True)
@click.option("-weights_path", "-wp",
              type=click.Path(resolve_path=True, dir_okay=False),
              help="Path to model weights")
@click.option("-result_filename", "-rf",
              type=click.Path(resolve_path=True, dir_okay=False),
              help="Path to model weights",
              default="result")
@click.option("-result_dir", "-rd",
              type=click.Path(resolve_path=True, dir_okay=True),
              help="Path to model weights",
              default="result")
@click.option("-numworkers", "-nw",
              type=click.Path(resolve_path=True, dir_okay=True),
              help="Path to model weights")
def main(path, weights_path, result_filename,
         result_dir):
    """

    :param path:
    :param weights_path:
    :param result_filename:
    :param result_dir:
    :return:
    """
    classify(
        dataset_path=path,
        weights_path=weights_path,
        result_path=result_filename,
        result_dir=result_dir,
        batch_size=BATCH_SIZE,
        workers_num=NUM_WORKERS,
        predicted_classes=CLASSES,
        device=PREDICT_DEVICE,
        result_columns=RESULT_COLUMNS,
    )

import lightning.pytorch as L
from lightning.pytorch.callbacks import BasePredictionWriter


def ensure_prediction_writer(trainer: L.Trainer) -> None:
    """Raise if the trainer has no prediction writer callback, since predictions would be lost.

    Inference runs trainer.predict with return_predictions=False, so a BasePredictionWriter
    in trainer.callbacks (conf/trainer/inference.yaml) is the only thing that saves outputs.
    Without one, inference would finish without error and write nothing.
    """
    # [purbancz] Lightning's callback connector sets Trainer.callbacks but Trainer never declares it.
    if not any(isinstance(callback, BasePredictionWriter) for callback in trainer.callbacks):  # type: ignore[attr-defined]
        raise ValueError(
            "No BasePredictionWriter in trainer.callbacks; predictions would not be saved "
            "(see conf/trainer/inference.yaml)."
        )

import logging

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LIBRARY_LOGGERS = ("lightning", "lightning.pytorch", "lightning.fabric", "torchmetrics")


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
    else:
        logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)
    root.setLevel(level)

    for name in LIBRARY_LOGGERS:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(level)

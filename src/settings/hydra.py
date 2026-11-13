from settings.paths import PROJECT_ROOT

VERSION_BASE = "1.3"
CONFIG_PATH = str(PROJECT_ROOT / "conf")
CONFIG_NAME = "config"

# Only these may be instantiated from configs stored in checkpoints
ALLOWED_TARGET_PREFIXES = ("gut_former.", "torch.optim.", "torch.nn.")

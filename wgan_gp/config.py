from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"

BASELINE_MODEL_DIR = PROJECT_ROOT / "baseline" / "models"
PREPROCESSOR_PATH = BASELINE_MODEL_DIR / "preprocessor.pkl"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
SYNTHETIC_DIR = OUTPUT_DIR / "synthetic"


# --------------------------------------------------
# DATA SETTINGS
# --------------------------------------------------

TARGET_COLUMN = "label"
ATTACK_CATEGORY_COLUMN = "attack_cat"

DROP_COLUMNS = ["id", "attack_cat", "label"]

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# Pilot category; can be changed after team confirmation.
TARGET_ATTACK_CATEGORY = "Shellcode"


# --------------------------------------------------
# WGAN-GP SETTINGS
# --------------------------------------------------

LATENT_DIM = 64

BATCH_SIZE = 64
EPOCHS = 300

LEARNING_RATE = 0.0001
CRITIC_STEPS = 5
GRADIENT_PENALTY_WEIGHT = 10.0

# Generator and Critic hidden layer sizes
HIDDEN_DIMS = (256, 128)

# Initial sample-generation target.
# We will finalize this after inspecting training results.
NUM_SYNTHETIC_SAMPLES = 1000


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def create_output_directories():
    """Create directories used to save training outputs."""

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

# Paths
CONFIG_PATH = "config/model.yaml"
TRAIN_DIR = "data/train/train"
VAL_DIR = "data/test/test"
TEST_DIR = "data/test/test"
SUBMISSION_DIR = "data/submission/"

# logic consts
ENTITY_MAPPING = {-1: "Team_A", 0: "Ball", 1: "Team_B"}

REL_TEAMMATE = 0
REL_OPPONENT = 1
REL_PLAYER_TO_BALL = 2
REL_BALL_TO_PLAYER = 3
REL_SELF = 4
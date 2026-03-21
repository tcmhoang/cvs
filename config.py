import os


BASE_DIR = os.getcwd()
CACHE_DIR = os.path.join(BASE_DIR, "cache")
TRAIN_DIR = os.path.join(CACHE_DIR, "train")
TEST_DIR = os.path.join(CACHE_DIR, "test")
EVAL_DIR = os.path.join(CACHE_DIR, "eval")

WANDB_PRJ = "Retrieval-GeM-DINOv2"

EMBEDDINGS = "embeddings.zst"
CSV_EXP = "results.csv"

TEST_PERC = 0.1
EVAL_PERC = 0.1

EMBEDDING_DIM = 384
GEM_P = 3.0
BATCH_SZ = 16
TEST_BATCH_SZ = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
TRIPLET_MARGIN = 0.5
NUM_WORKERS = 24

IMAGE_SIZE = 256
CROP_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

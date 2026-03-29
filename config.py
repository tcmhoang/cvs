import os


BASE_DIR = os.getcwd()
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUT_DIR = os.path.join(BASE_DIR, "out")
TRAIN_DIR = os.path.join(CACHE_DIR, "train")
TEST_DIR = os.path.join(CACHE_DIR, "test")
EVAL_DIR = os.path.join(CACHE_DIR, "eval")

MODEL_PATH = os.path.join(OUT_DIR, "model.pth")
RETRIEVAL_RES_PATH = os.path.join(OUT_DIR, "retrieval_results.csv")
EMBEDDING_PATH = os.path.join(OUT_DIR, "seed_embeddings.zst")

WANDB_PRJ = "Retrieval-GeM-DINOv2"

EMBEDDINGS = "embeddings.zst"
CSV_EXP = "results.csv"

TEST_PERC = 0.1
EVAL_PERC = 0.1

KTOP = 5
EMBEDDING_DIM = 384
GEM_P = 3.0
BATCH_SZ = 64
TEST_BATCH_SZ = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
TRIPLET_MARGIN = 0.5
NUM_WORKERS = 24

IMAGE_SZ = 256
CROP_SZ = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

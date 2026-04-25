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


def get_model_path(ins: str) -> str:
    return os.path.join(OUT_DIR, f"model-{ins}.pth")


def get_rev_res_path(ins: str) -> str:
    return os.path.join(OUT_DIR, f"retrieval_results-{ins}.csv")


def get_emb_path(ins: str) -> str:
    return os.path.join(OUT_DIR, f"seed_embeddings-{ins}.zst")


WANDB_PRJ = "Retrieval-GeM-DINOv2"

TEST_PERC = 0.1
EVAL_PERC = 0.1

KTOP = 5
EMBEDDING_DIM = 384
GEM_P = 3.0
BATCH_SZ = 32
TEST_BATCH_SZ = 32
EPOCHS = 50
WARM_UP_PERC = 0.1
WARM_UP_FACTOR = 0.1
LEARNING_RATE = 5e-6
NUM_WORKERS = 24

PA_A = 32
PA_MARGIN = 0.1

AQE_K = 3
AQE_ALPHA = 0.7


DETERMINISTIC = True
SEED_VAL = 7

IMAGE_SZ = 384
CROP_SZ = 336
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

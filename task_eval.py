from typing import cast

import torch

import config
import dataset
import evaluate
import model
import visualize
import wandb
from main import set_seed
from proc import Logger


def main():
    wandb.init(
        project="Retrieval-DINOv2",
        config={
            "architecture": "DINOv2_vits14 + GeM + MLP",
            "ref": input("instance? "),
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SZ,
            "learning_rate": config.LEARNING_RATE,
            "deterministic": config.DETERMINISTIC,
            "seed": config.SEED_VAL,
            "aqe": config.AQE_K,
        },
    )

    if config.DETERMINISTIC:
        set_seed(config.SEED_VAL)
        pass

    logger = cast(Logger, wandb)

    test_set = dataset.get_img_set(
        config.TEST_DIR,
        dataset.get_img_test_transform(
            config.IMAGE_SZ,
            config.CROP_SZ,
            config.NORMALIZE_MEAN,
            config.NORMALIZE_STD,
        ),
    )

    print("CUDA CHECK")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("RETRIEVE LOADER")
    test_loader = dataset.get_test_loaders(
        (test_set, config.TEST_BATCH_SZ),
        config.NUM_WORKERS,
    )

    print("LOAD MODEL")
    m = model.RetrievalNet(
        model.io_get_model(device), config.GEM_P, embeding_dim=config.EMBEDDING_DIM
    ).to(device)

    m.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))

    print("EVAL")
    features, labels = evaluate.extract_features(m, test_loader, device)
    features = evaluate.apply_aqe(features, k_aqe=config.AQE_K, a=config.AQE_ALPHA)

    r1, rk = evaluate.rank(features, labels, config.KTOP)
    map_score = evaluate.map(features, labels)

    print(f"Rank 1: {r1:.4f}")
    print(f"Rank {config.KTOP}: {rk:.4f}")
    print(f"mAP: {map_score:.4f}")

    logger.log(
        {
            "Rank-1 Accuracy": r1,
            f"Rank-{config.KTOP} Accuracy": rk,
            "mAP Score": map_score,
        }
    )

    evaluate.io_save(features, labels, config.EMBEDDING_PATH, config.OUT_DIR)

    evaluate.io_report_csv(
        features, labels, config.RETRIEVAL_RES_PATH, config.OUT_DIR, config.KTOP
    )

    visualize.plot_and_log_tsne(features, labels, logger)

    wandb.finish()

    pass


if __name__ == "__main__":
    main()

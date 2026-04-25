import os
import random
from typing import cast

import numpy as np
import torch

import config
import dataset
import dio
import evaluate
import model
import train
import visualize
import wandb
from proc import Logger
from datetime import datetime, timezone


def main():

    run = wandb.init(
        project="Retrieval-DINOv2",
        config={
            "architecture": "DINOv2_vits14 + GeM + MLP",
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SZ,
            "msk": config.MS_K,
            "learning_rate": config.LEARNING_RATE,
            "deterministic": config.DETERMINISTIC,
            "seed": config.SEED_VAL,
            "aqe": config.AQE_K,
        },
    )

    run_ins = run.name or datetime.now(timezone.utc).isoformat()

    logger = cast(Logger, wandb)

    if config.DETERMINISTIC:
        set_seed(config.SEED_VAL)
        pass

    print("CUDA CHECK")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("POP DS")
    failed_populateds = dio.prepare(
        {"lzupsd": None, "vsd": None}, config.CACHE_DIR, config.TRAIN_DIR
    )

    if len(failed_populateds) != 0:
        print("Failed to prepare", " ".join(failed_populateds))

    print("CATS DS")
    dio.cats(
        config.TEST_PERC,
        config.EVAL_PERC,
        (config.TRAIN_DIR, config.EVAL_DIR, config.TEST_DIR),
    )

    print("PREPARE DS")
    train_set, test_set = dataset.get_img_sets(
        (
            config.TRAIN_DIR,
            dataset.get_img_train_transform(
                config.CROP_SZ,
                config.NORMALIZE_MEAN,
                config.NORMALIZE_STD,
            ),
        ),
        (
            config.TEST_DIR,
            dataset.get_img_test_transform(
                config.IMAGE_SZ,
                config.CROP_SZ,
                config.NORMALIZE_MEAN,
                config.NORMALIZE_STD,
            ),
        ),
    )

    test_loader = dataset.get_test_loaders(
        (test_set, config.TEST_BATCH_SZ),
        config.NUM_WORKERS,
    )

    print("INIT MODEL")
    m = model.RetrievalNet(
        model.io_get_model(device), config.GEM_P, embeding_dim=config.EMBEDDING_DIM
    ).to(device)

    print("TRAIN")
    m = train.retrieval_model(
        m=m,
        train_dataset=train_set,
        device=device,
        logger=logger,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SZ,
        lr=config.LEARNING_RATE,
    )

    dio.create_dir(config.OUT_DIR)
    torch.save(
        m.state_dict(),
        config.get_model_path(run_ins),
    )

    print("EVAL")

    features, labels = evaluate.extract_features(m, test_loader, device)
    features = evaluate.apply_aqe(features, k_aqe=config.AQE_K, a=config.AQE_ALPHA)

    r1, rk = evaluate.rank(features, labels, config.KTOP)
    map_score = evaluate.map(features, labels)
    sil_score = evaluate.silhouette(features, labels)
    knn_acc = evaluate.knn(features, labels) * 100

    print(f"Rank 1: {r1:.4f}")
    print(f"Rank {config.KTOP}: {rk:.4f}")
    print(f"mAP: {map_score:.4f}")
    print(f"Silhouette: {sil_score:.4f}")
    print(f"k-NN Accuracy (k=5): {knn_acc:.2f}%")

    logger.log(
        {
            "Rank-1 Accuracy": r1,
            f"Rank-{config.KTOP} Accuracy": rk,
            "mAP Score": map_score,
            "sil": sil_score,
            "k-NN": knn_acc,
        }
    )

    evaluate.io_save(features, labels, config.get_emb_path(run_ins), config.OUT_DIR)

    evaluate.io_report_csv(
        features, labels, config.get_rev_res_path(run_ins), config.OUT_DIR, config.KTOP
    )

    visualize.plot_and_log_tsne(features, labels, logger)
    visualize.plot_umap(features, labels, logger)

    wandb.finish()

    pass


def set_seed(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()

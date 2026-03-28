import torch
import dio
import dataset
import config
import model
import train


def main():

    print("CUDA CHECK")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("POP DS")
    # TODO: Load from the nw src
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
                config.IMAGE_SZ,
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
        model.io_get_model(device), embeding_dim=config.EMBEDDING_DIM
    )

    print("TRAIN")
    m = train.retrieval_model(
        m=m,
        train_dataset=train_set,
        device=device,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SZ,
        lr=config.LEARNING_RATE,
        margin=config.TRIPLET_MARGIN,
    )

    print("EVAL")


if __name__ == "__main__":
    main()

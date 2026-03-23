import torch
import dataset
import config


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    failed_populateds = dataset.io_prepare(
        {"lzupsd": None, "vsd": None}, config.CACHE_DIR, config.TRAIN_DIR
    )

    if len(failed_populateds) != 0:
        print("Failed to prepare", " ".join(failed_populateds))

    dataset.io_cats(
        config.TEST_PERC,
        config.EVAL_PERC,
        (config.TRAIN_DIR, config.EVAL_DIR, config.TEST_DIR),
    )


if __name__ == "__main__":
    main()

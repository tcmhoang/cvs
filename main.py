import torch
import dio
import config


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    failed_populateds = dio.prepare(
        {"lzupsd": None, "vsd": None}, config.CACHE_DIR, config.TRAIN_DIR
    )

    if len(failed_populateds) != 0:
        print("Failed to prepare", " ".join(failed_populateds))

    dio.cats(
        config.TEST_PERC,
        config.EVAL_PERC,
        (config.TRAIN_DIR, config.EVAL_DIR, config.TEST_DIR),
    )


if __name__ == "__main__":
    main()

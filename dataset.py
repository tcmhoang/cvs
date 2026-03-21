import math
from typing import Callable, List, Optional, Tuple, cast
import os
import random
import functools
from itertools import filterfalse
import shutil


def io_create(*paths: str) -> None:
    for p in paths:
        try:
            if not os.path.exists(p):
                os.mkdir(p)
        except Exception:
            print(f"Failed to create {p}")


def io_prepare(
    reldest_with_srchandler: dict[str, Optional[Callable[[str], bool]]],
    cache_dir: str,
    train_dir: str,
) -> List[str]:

    if not all(map(os.path.exists, [cache_dir, train_dir])):
        io_create(cache_dir, train_dir)

    def populate(rel_w_path: Tuple[str, str]):
        mb_resolver = reldest_with_srchandler.get(rel_w_path[0])
        if mb_resolver is None:
            return rel_w_path[0]
        return "" if mb_resolver(train_dir) else rel_w_path[0]

    return list(
        filter(
            lambda x: len(x) != 0,
            map(
                populate,
                filterfalse(
                    lambda rel_w_path: os.path.exists(rel_w_path[1]),
                    map(
                        lambda reldest: (reldest, os.path.join(cache_dir, reldest)),
                        reldest_with_srchandler,
                    ),
                ),
            ),
        )
    )


def io_cats(test_perc: float, eval_perc: float, train_eval_test: Tuple[str, str, str]):
    if not all(map(os.path.exists, train_eval_test)):
        io_create(*train_eval_test)

    if test_perc + eval_perc > 0.3 or test_perc <= 0 or eval_perc <= 0:
        raise Exception(
            "Sum of Test and Evaluation's Sample percentage shoud not be larger than 30% and both must positive"
        )

    train_path, eval_path, test_path = train_eval_test
    target_dirs = set(os.listdir(train_path)).difference(
        {e for e in os.listdir(eval_path) + os.listdir(test_path)}
    )

    if len(target_dirs) == 0:
        return

    target_paths = map(lambda dir: os.path.join(train_path, dir), target_dirs)

    tot_perc = test_perc + eval_perc

    frac_test_eval = test_perc / eval_perc

    def extract(path: str, perc: float) -> List[str]:
        files = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
        random.shuffle(files)
        return [file for file in files[: max(math.floor(len(files) * perc), 2)]]

    test_files, eval_files = functools.reduce(
        lambda acc, tups: (acc[0] + tups[0], acc[1] + tups[1]),
        map(
            lambda files: (
                files[: max(1, math.floor(len(files) * frac_test_eval))],
                files[max(1, math.floor(len(files) * frac_test_eval)) :],
            ),
            map(lambda path: extract(path, tot_perc), target_paths),
        ),
        cast(Tuple[List[str], List[str]], ([], [])),
    )

    for f in test_files:
        shutil.move(f, os.path.join(test_path, os.path.dirname(f)))

    for f in eval_files:
        shutil.move(f, os.path.join(eval_path, os.path.dirname(f)))

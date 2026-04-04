import functools
import math
import os
import random
import shutil
from itertools import filterfalse
from typing import Callable, List, Optional, Tuple, cast


def create_dir(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
        pass
    return


def appendls(ls: List, lsp: List) -> List:
    ls.append(lsp)
    return ls


def prepare(
    reldest_with_srchandler: dict[str, Optional[Callable[[str], bool]]],
    cache_dir: str,
    train_dir: str,
) -> List[str]:

    if not all(map(os.path.exists, [cache_dir, train_dir])):
        create_dir(cache_dir, train_dir)

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


def cats(
    test_perc: float, eval_perc: float, train_eval_test: Tuple[str, str, str]
) -> None:
    if not all(map(os.path.exists, train_eval_test)):
        create_dir(*train_eval_test)

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

    frac_test_eval = test_perc / eval_perc * 0.5

    def go_split(path: str, perc: float) -> Tuple[List[str], List[str]]:
        files = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
        sample_size = max(math.floor(len(files) * perc), 2)
        sample = random.sample(files, sample_size)
        portion_size = math.floor(frac_test_eval * sample_size)
        return sample[:portion_size], sample[portion_size:]

    test_files, eval_files = functools.reduce(
        lambda acc, tups: (appendls(acc[0], tups[0]), appendls(acc[1], tups[1])),
        map(lambda path: go_split(path, tot_perc), target_paths),
        cast(Tuple[List[str], List[str]], ([], [])),
    )

    for files, path in [(test_files, test_path), (eval_files, eval_path)]:
        for f in files:
            clss_path = os.path.join(path, os.path.basename(os.path.dirname(f)))
            os.makedirs(clss_path, exist_ok=True)
            shutil.move(
                f,
                os.path.join(clss_path, os.path.basename(f)),
            )
            pass
        pass
    pass

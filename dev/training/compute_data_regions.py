import functools
import itertools
import logging
import multiprocessing
from typing import List

import tqdm

import core
import files_management as fm
import prepostprocessing as ppp


logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)


def compute_data_regions(
    pages: List[fm.PageMeta],
    distance_thresholds: List[float],
    minimum_depth: int,
    max_tags_per_gnode: int,
):
    n_runs = len(pages) * len(distance_thresholds)
    logging.info("Number of combinations: {}".format(n_runs))

    for th in tqdm.tqdm(distance_thresholds, desc="thresholds"):
        run_th = functools.partial(
            ppp.precompute_data_regions,
            threshold=th,
            minimum_depth=minimum_depth,
            max_tags_per_gnode=max_tags_per_gnode,
        )
        with multiprocessing.Pool() as pool:
            imap = pool.imap(run_th, pages)
            list(tqdm.tqdm(imap, total=len(pages), desc="pages"))


def main():
    # only get the annotated ones
    pages = {
        page_id: page_meta
        for page_id, page_meta in fm.PageMeta.get_all().items()
        if page_meta.distances_pkl.exists()
    }
    logging.info("Number of pages: %d.", len(pages))

    # todo add depth to data region  -->  can easily simulate minimum depth wo/ rerunning
    # todo make it possible to call up to finding data region only
    # distance_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
    distance_thresholds = [th / 100 for th in range(5, 50 + 1)]
    logging.info("Number of threshold: %d.", len(distance_thresholds))

    compute_data_regions(list(pages.values()), distance_thresholds, 3, 10)


if __name__ == "__main__":
    main()
    fm.cleanup_pages_meta_lock()

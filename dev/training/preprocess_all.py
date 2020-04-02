import functools
import logging
import multiprocessing
from typing import List

import tqdm

import core
import files_management as fm
import prepostprocessing as ppp

COMP_DIST_MAX_TAG_PER_GNODE = 15
COMP_DISTANCES_MIN_DEPTH = 0

MINIMUM_DEPTH = 3
MAX_TAGS_PER_GNODE = 10

N_PROCESSES = None


logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s"
)

logging.info("n available processes: %d", multiprocessing.cpu_count())


def download_all_pages(pages_metas):
    pages_metas = sorted(pages_metas.values(), key=lambda x: x.page_id)
    with multiprocessing.Pool(N_PROCESSES) as pool:
        imap = pool.imap(ppp.download_raw, pages_metas)
        list(tqdm.tqdm(imap, total=len(pages_metas)))


def cleanup_all_pages(pages_metas):
    pages_metas = sorted(pages_metas.values(), key=lambda x: x.page_id)
    with multiprocessing.Pool(N_PROCESSES) as pool:
        imap = pool.imap(ppp.cleanup_html, pages_metas)
        list(tqdm.tqdm(imap, total=len(pages_metas)))


def compute_all_distances(pages_metas):
    pages_metas = sorted(pages_metas.values(), key=lambda x: x.page_id)
    precompute_distances = functools.partial(
        ppp.precompute_distances,
        minimum_depth=COMP_DISTANCES_MIN_DEPTH,
        max_tag_per_gnode=COMP_DIST_MAX_TAG_PER_GNODE,
        force_override=False,
    )
    with multiprocessing.Pool(N_PROCESSES) as pool:
        imap = pool.imap(precompute_distances, pages_metas)
        list(tqdm.tqdm(imap, total=len(pages_metas)))


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
        with multiprocessing.Pool(N_PROCESSES) as pool:
            imap = pool.imap(run_th, pages)
            list(tqdm.tqdm(imap, total=len(pages), desc="pages"))


def compute_data_records(
    pages: List[fm.PageMeta],
    distance_thresholds: List[float],  # will only consider cases where all dist th are the same
    max_tags_per_gnode: int,
):
    n_runs = len(pages) * len(distance_thresholds)
    logging.info("Number of combinations: {}".format(n_runs))

    for th in tqdm.tqdm(distance_thresholds, desc="thresholds"):
        run_th = functools.partial(
            ppp.precompute_data_records,
            thresholds=core.MDREditDistanceThresholds.all_equal(th),
            max_tags_per_gnode=max_tags_per_gnode,
        )
        with multiprocessing.Pool(N_PROCESSES) as pool:
            pool.map(run_th, pages)


def main():
    # only get the annotated ones
    all_labeled_pages = {
        page_id: page_meta
        for page_id, page_meta in fm.PageMeta.get_all().items()
        if page_meta.n_data_records is not None
    }
    logging.info("Number of labeled pages: %d.", len(all_labeled_pages))
    download_all_pages(all_labeled_pages)

    all_downloaded_pages = {
        page_id: page_meta
        for page_id, page_meta in fm.PageMeta.get_all().items()
        if page_meta.raw_html.exists()
    }
    logging.info("Number of downloaded pages: %d.", len(all_downloaded_pages))
    cleanup_all_pages(all_downloaded_pages)

    all_cleaned_pages = {
        page_id: page_meta
        for page_id, page_meta in fm.PageMeta.get_all().items()
        if page_meta.preprocessed_html.exists()
    }
    logging.info("Number of preprocessed pages: %d.", len(all_cleaned_pages))
    compute_all_distances(all_cleaned_pages)

    pages_with_distance = {
        page_id: page_meta
        for page_id, page_meta in fm.PageMeta.get_all().items()
        if page_meta.distances_pkl.exists()
    }
    logging.info("Number of pages with distance: %d.", len(pages_with_distance))
    distance_thresholds = [th / 100 for th in range(5, 50 + 1)]
    logging.info("Number of threshold: %d.", len(distance_thresholds))
    compute_data_regions(
        list(pages_with_distance.values()), distance_thresholds, MINIMUM_DEPTH, MAX_TAGS_PER_GNODE
    )

    pages_with_distance_and_all_th = {
        page_id: page_meta
        for page_id, page_meta in fm.PageMeta.get_all().items()
        if page_meta.distances_pkl.exists()
        and all(page_meta.data_regions_pkl(th, MAX_TAGS_PER_GNODE) for th in distance_thresholds)
    }
    logging.info(
        "Number of pages to computed data records: %d.", len(pages_with_distance_and_all_th)
    )
    compute_data_records(
        list(pages_with_distance_and_all_th.values()), distance_thresholds, MAX_TAGS_PER_GNODE
    )


if __name__ == "__main__":
    main()
    from files_management import cleanup_pages_meta_lock

    cleanup_pages_meta_lock()

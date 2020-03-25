import functools
import logging
import multiprocessing

import tqdm

import core
import files_management as fm
import prepostprocessing as ppp


logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)


def compute_data_regions(pages, distance_thresholds):
    pass


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
    distance_thresholds = []

    compute_data_regions(pages, distance_thresholds)


if __name__ == "__main__":
    main()
    fm.cleanup_pages_meta_lock()

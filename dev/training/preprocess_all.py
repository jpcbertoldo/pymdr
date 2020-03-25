import functools
import logging
import multiprocessing

import tqdm

import core
import files_management as fm
import prepostprocessing as ppp


logging.basicConfig(level=logging.INFO)


def download_all_pages(pages_metas):
    pages_metas = sorted(pages_metas.values(), key=lambda x: x.page_id)
    with multiprocessing.Pool() as pool:
        imap = pool.imap(ppp.download_raw, pages_metas)
        list(tqdm.tqdm(imap, total=len(pages_metas)))


def cleanup_all_pages(pages_metas):
    pages_metas = sorted(pages_metas.values(), key=lambda x: x.page_id)
    with multiprocessing.Pool() as pool:
        imap = pool.imap(ppp.cleanup_html, pages_metas)
        list(tqdm.tqdm(imap, total=len(pages_metas)))


def compute_all_distances(pages_metas):
    pages_metas = sorted(pages_metas.values(), key=lambda x: x.page_id)
    precompute_distances = functools.partial(
        ppp.precompute_distances, minimum_depth=0, max_tag_per_gnode=15, force_override=False
    )
    with multiprocessing.Pool() as pool:
        imap = pool.imap(precompute_distances, pages_metas)
        list(tqdm.tqdm(imap, total=len(pages_metas)))


def main():
    # only get the annotated ones
    all_labeled_pages = {
        page_id: page_meta
        for page_id, page_meta in fm.PageMeta.get_all().items()
        if page_meta.n_data_records is not None
    }
    logging.info("Number of labeled pages: %d.", len(all_labeled_pages))
    # download_all_pages(all_labeled_pages)

    all_downloaded_pages = {
        page_id: page_meta
        for page_id, page_meta in all_labeled_pages.items()
        if page_meta.raw_html.exists()
    }
    logging.info("Number of downloaded pages: %d.", len(all_downloaded_pages))
    # cleanup_all_pages(all_downloaded_pages)

    all_cleaned_pages = {
        page_id: page_meta
        for page_id, page_meta in all_labeled_pages.items()
        if page_meta.preprocessed_html.exists()
    }
    logging.info("Number of preprocessed pages: %d.", len(all_cleaned_pages))
    compute_all_distances(all_downloaded_pages)


if __name__ == "__main__":
    main()
    from files_management import cleanup_pages_meta_lock

    cleanup_pages_meta_lock()
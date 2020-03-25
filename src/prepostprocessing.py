import datetime
import logging
import urllib
import urllib.request
import urllib.response

import retrying

from src.core import MDR
from src.files_management import PageMeta

logging.basicConfig(level=logging.INFO)

SEC = 1000  # in ms

all_metas = PageMeta.get_all()


def download_raw(page_meta: PageMeta, force_override: bool = False) -> None:
    logging.info(
        "`%s` called for page_id=%s", download_raw.__name__, page_meta.page_id
    )
    exists = page_meta.raw_html.exists()

    if exists:
        logging.info(
            "Raw page has already been downloaded. page_id=%s",
            page_meta.page_id,
        )
        if force_override:
            logging.info(
                "It will be overwritten. page_id=%s", page_meta.page_id
            )
        else:
            logging.info("Operation skipped. page_id=%s", page_meta.page_id)
            return
    else:
        logging.info(
            "Raw page will be downloaded. page_id=%s", page_meta.page_id
        )

    @retrying.retry(
        stop_max_attempt_number=10,
        wait_exponential_multiplier=SEC,
        wait_exponential_max=10 * SEC,
        wrap_exception=True,
    )
    def call_url():
        logging.info("Requesting the page...  page_id=%s", page_meta.page_id)
        response = urllib.request.urlopen(page_meta.url, timeout=10)
        page_binary = response.read()
        return page_binary

    try:
        page = call_url()
    except retrying.RetryError:
        logging.warning(
            "Failed download the page, returning. page_id=%s",
            page_meta.page_id,
        )
        return

    logging.info("Writing down the file. page_id=%s", page_meta.page_id)
    with page_meta.raw_html.open(mode="wb") as f:
        f.write(page)

    logging.info(
        "Saving download time in metadata file. page_id=%s", page_meta.page_id
    )
    now = datetime.datetime.now()
    page_meta.persist_download_datetime(now)
    logging.info("Done")


def cleanup_html(page_meta: PageMeta) -> None:
    pass


def persist_named_tags_html(page_meta: PageMeta, mdr: MDR) -> None:
    # todo remember to also markup the data records
    pass


def color_html(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_data_regions(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_data_records(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_mdr_parameters(page_meta: PageMeta, mdr: MDR) -> None:
    pass

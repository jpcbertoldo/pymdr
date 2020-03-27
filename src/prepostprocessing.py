import datetime
import logging
import urllib
import urllib.request
import urllib.response

import lxml
import lxml.etree
import lxml.html
import retrying

import core
from core import MDR
from files_management import PageMeta, open_html_document


logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s"
)

SEC = 1000  # in ms

all_metas = PageMeta.get_all()


def download_raw(page_meta: PageMeta, force_override: bool = False) -> None:
    logging.info("page_id=%s", page_meta.page_id)
    exists = page_meta.raw_html.exists()

    if exists:
        logging.info(
            "Raw page has already been downloaded. page_id=%s", page_meta.page_id,
        )
        if force_override:
            logging.info("It will be overwritten. page_id=%s", page_meta.page_id)
        else:
            logging.info("Operation skipped. page_id=%s", page_meta.page_id)
            return
    else:
        logging.info("Raw page will be downloaded. page_id=%s", page_meta.page_id)

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
            "Failed download the page, returning. page_id=%s", page_meta.page_id,
        )
        return

    logging.info("Writing down the file. page_id=%s", page_meta.page_id)
    PageMeta.persist_html(page_meta.raw_html, page)

    logging.info("Saving download time in metadata file. page_id=%s", page_meta.page_id)
    now = datetime.datetime.now()
    page_meta.persist_download_datetime(now)
    logging.info("Done. page_id=%s", page_meta.page_id)


def cleanup_html(page_meta: PageMeta, force_override: bool = False) -> None:
    logging.info("page_id=%s", page_meta.page_id)
    exists = page_meta.preprocessed_html.exists()

    if exists:
        logging.info(
            "Page has already been preprocessed. page_id=%s", page_meta.page_id,
        )
        if force_override:
            logging.info("It will be overwritten. page_id=%s", page_meta.page_id)
        else:
            logging.info("Operation skipped. page_id=%s", page_meta.page_id)
            return
    else:
        logging.info("Raw page will be preprocessed. page_id=%s", page_meta.page_id)

    logging.info(
        "Opening raw html file by removing stuff. page_id=%s", page_meta.page_id,
    )
    doc = open_html_document(page_meta.raw_html, remove_stuff=True)
    logging.info(
        "Stripping <meta>, <script>, and <style> tags. page_id=%s", page_meta.page_id,
    )
    lxml.etree.strip_elements(doc, "script")
    lxml.etree.strip_elements(doc, "style")
    lxml.etree.strip_elements(doc, "meta")

    logging.info("Writing down the file. page_id=%s", page_meta.page_id)
    PageMeta.persist_html(page_meta.preprocessed_html, doc)

    logging.info("Done. page_id=%s", page_meta.page_id)


def precompute_distances(
    page_meta: PageMeta, minimum_depth, max_tag_per_gnode, force_override: bool = False
):
    logging.info("page_id=%s", page_meta.page_id)
    exists = page_meta.distances_pkl.exists()

    if exists:
        logging.info(
            "Distances have already been precomputed, checking parameters... page_id=%s",
            page_meta.page_id,
        )
        precomputed = page_meta.load_precomputed_distances()
        precomputed_minimum_depth = precomputed["minimum_depth"]
        precomputed_max_tag_per_gnode = precomputed["max_tag_per_gnode"]
        precomputed_was_more_restrictive = (
            precomputed_max_tag_per_gnode < max_tag_per_gnode
            or precomputed_minimum_depth > minimum_depth
        )
        if force_override:
            logging.info("It will be overwritten. page_id=%s", page_meta.page_id)
        elif precomputed_was_more_restrictive:
            logging.info(
                "The previously computed was more restrictive. It'll be overwritten. page_id=%s",
                page_meta.page_id,
            )
        else:
            logging.info("Operation skipped. page_id=%s", page_meta.page_id)
            return
    else:
        logging.info("The distances will be computed. page_id=%s", page_meta.page_id)

    assert page_meta.preprocessed_html.exists()
    logging.info("Opening precomputed html. page_id=%s", page_meta.page_id)
    doc = open_html_document(page_meta.preprocessed_html, remove_stuff=False)

    logging.info("Loading node namer. page_id=%s", page_meta.page_id)
    node_namer = core.NodeNamer()
    node_namer.load(doc)
    # todo persist named nodes here

    logging.info("Computing distances. page_id=%s", page_meta.page_id)
    distances = {}
    core.MDR.compute_distances(doc, distances, {}, node_namer, minimum_depth, max_tag_per_gnode)

    logging.info("Persisting distances. page_id=%s", page_meta.page_id)
    page_meta.persist_precomputed_distances(distances, minimum_depth, max_tag_per_gnode)

    logging.info("Done. page_id=%s", page_meta.page_id)


def precompute_data_regions(
    page_meta: PageMeta,
    threshold: float,
    minimum_depth: int,
    max_tags_per_gnode: int,
    force_override: bool = False,
):
    logging.info("page_id=%s", page_meta.page_id)
    assert page_meta.distances_pkl.exists(), "Distances have NOT been precomputed!"

    exists = page_meta.data_regions_pkl(threshold, max_tags_per_gnode).exists()

    if exists:
        logging.info(
            "The data regions have already been precomputed, checking parameters... page_id=%s th=%.2f max_tags=%d",
            page_meta.page_id,
            threshold,
            max_tags_per_gnode,
        )

        if force_override:
            logging.info(
                "It will be overwritten. page_id=%s th=%.2f max_tags=%d",
                page_meta.page_id,
                threshold,
                max_tags_per_gnode,
            )
        else:
            precomputed = page_meta.load_precomputed_data_regions(threshold, max_tags_per_gnode)
            precomputed_minimum_depth = precomputed["minimum_depth"]
            if precomputed_minimum_depth > minimum_depth:
                logging.info(
                    "The previously computed was more restrictive. It'll be overwritten. page_id=%s th=%.2f max_tags=%d",
                    page_meta.page_id,
                    threshold,
                    max_tags_per_gnode,
                )
            else:
                logging.info(
                    "Operation skipped. page_id=%s th=%.2f max_tags=%d",
                    page_meta.page_id,
                    threshold,
                    max_tags_per_gnode,
                )
                return
    else:
        logging.info(
            "The data regions will be computed. page_id=%s th=%.2f max_tags=%d",
            page_meta.page_id,
            threshold,
            max_tags_per_gnode,
        )

    if page_meta.named_nodes_html.exists():
        logging.info(
            "Loading the named nodes html. page_id=%s th=%.2f max_tags=%d",
            page_meta.page_id,
            threshold,
            max_tags_per_gnode,
        )
        root = page_meta.get_named_nodes_html_tree()
        node_namer = core.NodeNamer(for_loaded_file=True)
    else:
        logging.info(
            "Named nodes have NOT been saved, computing it. page_id=%s th=%.2f max_tags=%d",
            page_meta.page_id,
            threshold,
            max_tags_per_gnode,
        )
        root = page_meta.get_preprocessed_html_tree()
        node_namer = core.NodeNamer()
        node_namer.load(root)

        logging.info(
            "Named nodes will be saved. page_id=%s th=%.2f max_tags=%d",
            page_meta.page_id,
            threshold,
            max_tags_per_gnode,
        )
        PageMeta.persist_html(page_meta.named_nodes_html, root)

    logging.info(
        "Loading precomputed distances. page_id=%s th=%.2f max_tags=%d",
        page_meta.page_id,
        threshold,
        max_tags_per_gnode,
    )
    distances = page_meta.load_precomputed_distances()

    logging.info(
        "Starting to compute data regions. page_id=%s th=%.2f max_tags=%d",
        page_meta.page_id,
        threshold,
        max_tags_per_gnode,
    )
    data_regions = {}
    core.MDR.find_data_regions(
        root, node_namer, minimum_depth, distances, data_regions, threshold, max_tags_per_gnode
    )

    logging.info(
        "Persisting data regions. page_id=%s th=%.2f max_tags=%d",
        page_meta.page_id,
        threshold,
        max_tags_per_gnode,
    )
    page_meta.persist_precomputed_data_regions(
        data_regions, threshold, minimum_depth, max_tags_per_gnode
    )

    logging.info(
        "Done. page_id=%s th=%.2f max_tags=%d", page_meta.page_id, threshold, max_tags_per_gnode
    )


def persist_named_tags_html(page_meta: PageMeta, mdr: MDR) -> None:
    # todo remember to also markup the data records
    pass


def color_html(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_data_records(page_meta: PageMeta, mdr: MDR) -> None:
    pass


def persist_mdr_parameters(page_meta: PageMeta, mdr: MDR) -> None:
    pass

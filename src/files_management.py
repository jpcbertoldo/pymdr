"""
Module dependencies:
    all - {utils, core} -> files_management
"""

import datetime
import functools
import hashlib
import logging
import pathlib
import pickle
from typing import Dict, Optional, Union, List

import lxml
import lxml.etree
import lxml.html
import yaml
from oslo_concurrency import lockutils

import core
import utils

logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)


def open_html_document(filepath: pathlib.Path, remove_stuff: bool) -> lxml.html.HtmlElement:
    """
    todo(unittest)
    Returns:
        root of the html file
    """
    with filepath.open("r") as file:
        html_document = lxml.html.fromstring(
            html=lxml.etree.tostring(lxml.html.parse(file), method="html"),
            # todo (unittest) add test for comments? it has broken the code...
            parser=lxml.etree.HTMLParser(
                remove_comments=remove_stuff,
                remove_pis=remove_stuff,
                remove_blank_text=remove_stuff,
            ),
        )
    return html_document


def make_outputs_dir(in_dir: pathlib.Path):
    if isinstance(in_dir, str):
        in_dir = pathlib.Path(in_dir).absolute()

    outputs_dir_ = in_dir.joinpath("outputs").absolute()
    raw_htmls_dir_ = outputs_dir_.joinpath("raw_htmls").absolute()
    preprocessed_htmls_dir_ = outputs_dir_.joinpath("preprocessed_htmls")
    intermediate_results_dir_ = outputs_dir_.joinpath("intermediate_results").absolute()
    results_dir_ = outputs_dir_.joinpath("results").absolute()
    pages_meta_ = outputs_dir_.joinpath("pages-meta.yml").absolute()

    # create directories
    outputs_dir_.mkdir(parents=False, exist_ok=True)
    raw_htmls_dir_.mkdir(parents=False, exist_ok=True)
    preprocessed_htmls_dir_.mkdir(parents=False, exist_ok=True)
    intermediate_results_dir_.mkdir(parents=False, exist_ok=True)
    results_dir_.mkdir(parents=False, exist_ok=True)

    # create page's metadata file
    pages_meta_.touch(exist_ok=True)

    # todo: make .gitignore

    return (
        outputs_dir_,
        raw_htmls_dir_,
        preprocessed_htmls_dir_,
        intermediate_results_dir_,
        results_dir_,
        pages_meta_,
    )


outputs_parent_dir_ = utils.get_config_outputs_parent_dir()
logging.info("Outputs parent dir: %s", str(outputs_parent_dir_))

(
    outputs_dir,
    raw_htmls_dir,
    preprocessed_htmls_dir,
    intermediate_results_dir,
    results_dir,
    pages_meta,
) = make_outputs_dir(outputs_parent_dir_)
logging.info("Outputs dir: %s", str(outputs_dir))

lock_file_prefix = __file__[:-3]
synchronized = lockutils.synchronized_with_prefix(lock_file_prefix)
prefixed_cleanup = lockutils.remove_external_lock_file_with_prefix(lock_file_prefix)
pages_meta_lock_name = "pages_meta"
lock_path = str(outputs_dir)
cleanup_pages_meta_lock = functools.partial(
    prefixed_cleanup, name=pages_meta_lock_name, lock_path=lock_path
)


@synchronized(pages_meta_lock_name, external=True, fair=True, lock_path=lock_path)
def _read_metas_dict() -> dict:
    with pages_meta.open(mode="r") as f:
        metas_dict = yaml.load(f, Loader=yaml.FullLoader) or dict()
    return metas_dict


@synchronized(pages_meta_lock_name, external=True, fair=True, lock_path=lock_path)
def _write_metas_dict(metas: dict):
    with pages_meta.open(mode="w") as f:
        yaml.dump(metas, f, Dumper=yaml.SafeDumper)


class PageMeta(object):
    def __hash__(self):
        return hashlib.sha1(self.url.encode("utf-8")).digest()

    def __eq__(self, other):
        return hash(self) == hash(other)

    @staticmethod
    def _page_id(url: str) -> str:
        # todo change to hashlib and update pages-meta
        hash_digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return "-".join((hash_digest[:3], hash_digest[3:6], hash_digest[6:9]))

    @staticmethod
    def is_registered(url: str) -> bool:
        page_id = PageMeta._page_id(url)
        metas = _read_metas_dict()
        return page_id in metas.keys()

    @staticmethod
    def count() -> int:
        return len(_read_metas_dict())

    @staticmethod
    def get_all() -> Dict[str, "PageMeta"]:
        metas_dict = _read_metas_dict()
        all_metas = {
            page_id: PageMeta.from_dict(page_meta_dic)
            for page_id, page_meta_dic in metas_dict.items()
        }
        return all_metas

    @staticmethod
    def persist_html(html_path: pathlib.Path, doc: Union[bytes, lxml.html.HtmlElement]) -> None:
        with html_path.open("wb") as f:
            if isinstance(doc, bytes):
                f.write(doc)
            elif isinstance(doc, lxml.etree._Element):
                f.write(
                    lxml.etree.tostring(doc, encoding="utf-8", method="html", pretty_print=True)
                )
            else:
                raise TypeError("type `{}` of doc is not supported.".format(type(doc).__name__))

    @classmethod
    def register(cls, url: str, n_data_records: int) -> None:
        now = datetime.datetime.now()
        page_id = PageMeta._page_id(url)
        obj = cls(now, url, page_id, n_data_records, None)
        obj._persist()
        return obj

    @classmethod
    def from_meta_file_by_url(cls, url: str):
        meta_yaml = _read_metas_dict()
        page_id = PageMeta._page_id(url)
        assert page_id in meta_yaml.keys(), "Url has not been registered. url={}".format(url)
        dic = meta_yaml[page_id]
        return cls.from_dict(dic)

    @classmethod
    def from_meta_file_by_page_id(cls, page_id: str):
        meta_yaml = _read_metas_dict()
        assert page_id in meta_yaml.keys(), "Page id has not been registered. page_id={}".format(
            page_id
        )
        dic = meta_yaml[page_id]
        return cls.from_dict(dic)

    @classmethod
    def from_dict(cls, dic: dict):
        return cls(
            dic["date_time"],
            dic["url"],
            dic["page_id"],
            dic.get("n_data_records"),
            dic.get("download_datetime"),
        )

    def __init__(
        self,
        date_time: datetime.datetime,
        url: str,
        page_id: str,
        n_data_records: Optional[int],
        download_datetime: Optional[datetime.datetime],
    ):
        # todo make these private and make props
        # todo add download time_
        self.date_time = date_time
        self.url = url
        self.page_id = page_id
        self._n_data_records = n_data_records
        self._download_datetime = download_datetime

    @property
    def n_data_records(self) -> Optional[int]:
        return self._n_data_records

    @property
    def download_datetime(self) -> Optional[datetime.datetime]:
        return self._download_datetime

    @property
    def prefix(self):
        return "{date_time}---{page_id}-".format(
            date_time=self.date_time.strftime("%Y-%m-%d-%H%M"), page_id=self.page_id,
        )

    @property
    def raw_html(self) -> pathlib.Path:
        return raw_htmls_dir.joinpath(self.prefix + "raw.html").absolute()

    @property
    def preprocessed_html(self) -> pathlib.Path:
        return preprocessed_htmls_dir.joinpath(self.prefix + "preprocessed.html").absolute()

    @property
    def named_nodes_html(self) -> pathlib.Path:
        return preprocessed_htmls_dir.joinpath(self.prefix + "named_nodes.html").absolute()

    @property
    def distances_pkl(self) -> pathlib.Path:
        return intermediate_results_dir.joinpath(self.prefix + "distances.pkl").absolute()

    @property
    def colored_html(self) -> pathlib.Path:
        return results_dir.joinpath(self.prefix + "colored.html").absolute()

    @property
    def colored_graph(self) -> pathlib.Path:
        return results_dir.joinpath(self.prefix + "colored.pdf").absolute()

    def data_regions_pkl(self, threshold: float, max_tags_per_gnode: int) -> pathlib.Path:
        # todo(doc) only 2 decimal digits!!!!!
        return intermediate_results_dir.joinpath(
            self.prefix
            + "data_regions(th={:.2f},max_tags={}).pkl".format(threshold, max_tags_per_gnode)
        ).absolute()

    def data_records_pkl(
        self, thresholds: core.MDREditDistanceThresholds, max_tags_per_gnode: int
    ) -> pathlib.Path:
        # todo(doc) only 2 decimal digits!!!!!
        return results_dir.joinpath(
            self.prefix
            + "data_records(dr-th={:.2f},r1-th={:.2f},rn-th={:.2f},max_tags={}).pkl".format(
                thresholds.data_region,
                thresholds.find_records_1,
                thresholds.find_records_n,
                max_tags_per_gnode,
            )
        ).absolute()

    def _persist(self, is_new=True):
        meta_yaml = _read_metas_dict()
        assert (is_new and self.page_id not in meta_yaml) or (
            not is_new and self.page_id in meta_yaml
        ), "Url has already been registered. page_id={} url={}".format(self.page_id, self.url)
        meta_yaml[self.page_id] = self.to_dict()
        _write_metas_dict(meta_yaml)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "page_id": self.page_id,
            "date_time": self.date_time,
            "n_data_records": self._n_data_records,
            "download_datetime": self._download_datetime,
        }

    def get_raw_html_tree(self, remove_stuff: bool = False) -> lxml.html.HtmlElement:
        with self.raw_html.open(mode="r") as file:
            doc = lxml.html.fromstring(
                html=lxml.etree.tostring(lxml.html.parse(file), method="html"),
                # todo (unittest) add test for comments? it has broken the code...
                parser=lxml.etree.HTMLParser(
                    remove_comments=remove_stuff,
                    remove_pis=remove_stuff,
                    remove_blank_text=remove_stuff,
                ),
            )
        return doc

    def get_preprocessed_html_tree(self) -> lxml.html.HtmlElement:
        with self.preprocessed_html.open(mode="r") as file:
            doc = lxml.html.fromstring(
                html=lxml.etree.tostring(lxml.html.parse(file), method="html"),
            )
        return doc

    def get_named_nodes_html_tree(self) -> lxml.html.HtmlElement:
        with self.named_nodes_html.open(mode="r") as file:
            doc = lxml.html.fromstring(
                html=lxml.etree.tostring(lxml.html.parse(file), method="html"),
            )
        return doc

    def persist_precomputed_distances(
        self, dists: core.DISTANCES_DICT_FORMAT, minimum_depth: int, max_tag_per_gnode: int,
    ):
        dists["minimum_depth"] = minimum_depth
        dists["max_tag_per_gnode"] = max_tag_per_gnode
        with self.distances_pkl.open(mode="wb") as f:
            pickle.dump(dists, f)

    def load_precomputed_distances(self,) -> core.DISTANCES_DICT_FORMAT:
        assert self.distances_pkl.exists()
        with self.distances_pkl.open(mode="rb") as f:
            dists = pickle.load(f)
        return dists

    def persist_precomputed_data_regions(
        self,
        data_regions: core.DATA_REGION_DICT_FORMAT,
        distance_threshold: float,
        minimum_depth: int,
        max_tags_per_gnode: int,
    ):
        data_regions["distance_threshold"] = distance_threshold
        data_regions["minimum_depth"] = minimum_depth
        data_regions["max_tags_per_gnode"] = max_tags_per_gnode

        with self.data_regions_pkl(distance_threshold, max_tags_per_gnode).open(mode="wb") as f:
            pickle.dump(data_regions, f)

    def load_precomputed_data_regions(
        self, threshold: float, max_tags_per_gnode: int
    ) -> core.DATA_REGION_DICT_FORMAT:
        data_regions_pkl = self.data_regions_pkl(threshold, max_tags_per_gnode)
        assert data_regions_pkl.exists()
        with data_regions_pkl.open(mode="rb") as f:
            drs = pickle.load(f)
        return drs

    def persist_precomputed_data_records(
        self,
        data_records: core.DATA_RECORD_LIST,
        thresholds: core.MDREditDistanceThresholds,
        max_tags_per_gnode: int,
    ):
        with self.data_records_pkl(thresholds, max_tags_per_gnode).open(mode="wb") as f:
            pickle.dump(data_records, f)

    def load_precomputed_data_records(
        self, thresholds: core.MDREditDistanceThresholds, max_tags_per_gnode: int
    ) -> core.DATA_RECORD_LIST:
        data_records_pkl = self.data_records_pkl(thresholds, max_tags_per_gnode)
        assert data_records_pkl.exists()
        with data_records_pkl.open(mode="rb") as f:
            drs = pickle.load(f)
        return drs

    def persist_download_datetime(self, download_datetime: datetime.datetime) -> None:
        self._download_datetime = download_datetime
        self._persist(is_new=False)

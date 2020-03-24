import datetime
import os
import pathlib
import pickle
from typing import Dict, Optional, Union

import lxml
import lxml.etree
import lxml.html
import yaml


def open_html_document(
    directory: str = None, file: str = None, filepath: str = None
) -> lxml.html.HtmlElement:
    """
    todo(unittest)
    Returns:
        root of the html file
    """
    filepath = (
        os.path.join(os.path.abspath(directory), file)
        if filepath is None
        else filepath
    )
    with open(filepath, "r") as file:
        html_document = lxml.html.fromstring(
            html=lxml.etree.tostring(lxml.html.parse(file), method="html"),
            # todo (unittest) add test for comments? it has broken the code...
            parser=lxml.etree.HTMLParser(
                remove_comments=True, remove_pis=True, remove_blank_text=True
            ),
        )
    return html_document


def make_outputs_dir(in_dir=Union[str, pathlib.Path]):
    if isinstance(in_dir, str):
        in_dir = pathlib.Path(in_dir).absolute()

    outputs_dir_ = in_dir.joinpath("outputs").absolute()
    raw_htmls_dir_ = outputs_dir_.joinpath("raw_htmls").absolute()
    preprocessed_htmls_dir_ = outputs_dir_.joinpath("preprocessed_htmls")
    intermediate_results_dir_ = outputs_dir_.joinpath(
        "intermediate_results"
    ).absolute()
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


this_scripts_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(this_scripts_dir, ".."))

(
    outputs_dir,
    raw_htmls_dir,
    preprocessed_htmls_dir,
    intermediate_results_dir,
    results_dir,
    pages_meta,
) = make_outputs_dir(project_dir)


class PageMeta(object):
    @staticmethod
    def _page_id(url: str) -> str:
        number = "{:+09x}".format(hash(url))[-9:]
        return "-".join((number[:3], number[3:6], number[6:9]))

    @staticmethod
    def _get_metas_dict() -> dict:
        with pages_meta.open(mode="r") as f:
            metas_dict = yaml.load(f, Loader=yaml.FullLoader) or dict()
        return metas_dict

    @staticmethod
    def is_registered(url: str) -> bool:
        page_id = PageMeta._page_id(url)
        with pages_meta.open(mode="r") as f:
            meta_file = yaml.load(f, Loader=yaml.FullLoader) or dict()
        return page_id in meta_file.keys()

    @staticmethod
    def count() -> int:
        return len(PageMeta._get_metas_dict())

    @staticmethod
    def get_all() -> Dict[str, "PageMeta"]:
        metas_dict = PageMeta._get_metas_dict()
        all_metas = {
            page_id: PageMeta.from_dict(page_meta_dic)
            for page_id, page_meta_dic in metas_dict.items()
        }
        return all_metas

    @classmethod
    def register(cls, url: str, n_data_records: int):
        now = datetime.datetime.now()
        obj = cls(now, url, n_data_records)
        obj._persist()
        return obj

    @classmethod
    def from_meta_file(cls, url: str):
        with pages_meta.open("r") as f:
            meta_yaml = yaml.load(f, Loader=yaml.FullLoader) or dict()
        page_id = cls._page_id(url)
        assert (
            page_id in meta_yaml
        ), "Url has not been registered. url={}".format(url)
        dic = meta_yaml[page_id]
        return cls.from_dict(dic)

    @classmethod
    def from_dict(cls, dic: dict):
        return cls(dic["date_time"], dic["url"], dic["n_data_records"])

    def __init__(
        self, date_time: datetime.datetime, url: str, n_data_records: int
    ):
        # todo make these private and make props
        # todo add download time_
        self.date_time = date_time
        self.url = url
        self.n_data_records = n_data_records

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def page_id(self):
        return self._page_id(self.url)

    @property
    def prefix(self):
        return "{date_time}---{page_id}-".format(
            date_time=self.date_time.strftime("%Y-%m-%d-%H%M"),
            page_id=self.page_id,
        )

    @property
    def raw_html(self) -> pathlib.Path:
        return raw_htmls_dir.joinpath(self.prefix + "raw.html").absolute()

    @property
    def preprocessed_html(self) -> pathlib.Path:
        return preprocessed_htmls_dir.joinpath(
            self.prefix + "preprocessed.html"
        ).absolute()

    @property
    def distances_pkl(self) -> pathlib.Path:
        return intermediate_results_dir.joinpath(
            self.prefix + "distances.pkl"
        ).absolute()

    @property
    def colored_html(self) -> pathlib.Path:
        return results_dir.joinpath(self.prefix + "colored.html").absolute()

    @property
    def colored_graph(self) -> pathlib.Path:
        return results_dir.joinpath(self.prefix + "colored.pdf").absolute()

    def _persist(self):
        with pages_meta.open("r") as f:
            meta_yaml = yaml.load(f, Loader=yaml.FullLoader) or dict()
        assert (
            self.page_id not in meta_yaml
        ), "Url has already been registered. page_id={} url={}".format(
            self.page_id, self.url
        )
        meta_yaml[self.page_id] = self.to_dict()
        with pages_meta.open("w") as f:
            yaml.dump(meta_yaml, f)

    def to_dict(self):
        return {
            "url": self.url,
            "page_id": self.page_id,
            "date_time": self.date_time,
            "n_data_records": self.n_data_records,
        }

    def get_raw_html_tree(
        self, remove_stuff: bool = False
    ) -> lxml.html.HtmlElement:
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

    def persist_precomputed_distances(
        self, dists: Dict[str, Optional[Dict[int, Dict["GNodePair", float]]]]
    ):
        with self.distances_pkl.open(mode="wb") as f:
            pickle.dump(dists, f)

    def load_precomputed_distances(
        self,
    ) -> Dict[str, Optional[Dict[int, Dict["GNodePair", float]]]]:
        if not self.distances_pkl.exists():
            return {}
        with self.distances_pkl.open(mode="rb") as f:
            dists = pickle.load(f)
        return dists

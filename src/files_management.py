import datetime
import os
import pathlib
import pickle
from typing import Dict, Optional

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


this_scripts_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(this_scripts_dir, ".."))
outputs_dir = pathlib.Path(os.path.join(project_dir, "outputs"))
raw_htmls_dir = outputs_dir.joinpath("raw_htmls").absolute()
preprocessed_htmls_dir = outputs_dir.joinpath("preprocessed_htmls")
intermediate_results_dir = outputs_dir.joinpath(
    "intermediate_results"
).absolute()
results_dir = outputs_dir.joinpath("results").absolute()
pages_meta = outputs_dir.joinpath("pages-meta.yml").absolute()

# create directories
outputs_dir.mkdir(parents=False, exist_ok=True)
raw_htmls_dir.mkdir(parents=False, exist_ok=True)
preprocessed_htmls_dir.mkdir(parents=False, exist_ok=True)
intermediate_results_dir.mkdir(parents=False, exist_ok=True)
results_dir.mkdir(parents=False, exist_ok=True)

# create page's metadata file
pages_meta.touch(exist_ok=True)


class PageMeta(object):
    @staticmethod
    def _page_id(url):
        number = "{:+09x}".format(hash(url))[-9:]
        return "-".join((number[:3], number[3:6], number[6:9]))

    @staticmethod
    def is_registered(url):
        page_id = PageMeta._page_id(url)
        with pages_meta.open(mode="r") as f:
            meta_file = yaml.load(f, Loader=yaml.FullLoader) or dict()
        return page_id in meta_file.keys()

    @staticmethod
    def count():
        with pages_meta.open(mode="r") as f:
            meta_file = yaml.load(f, Loader=yaml.FullLoader) or dict()
        return len(meta_file)

    def __init__(self, date_time, url, n_data_records):
        # todo make these private and make props
        self.date_time = date_time
        self.url = url
        self.n_data_records = n_data_records

    @classmethod
    def register(cls, url, n_data_records):
        now = datetime.datetime.now()
        obj = cls(now, url, n_data_records)
        obj._persist()
        return obj

    @classmethod
    def from_meta_file(cls, url):
        with pages_meta.open("r") as f:
            meta_yaml = yaml.load(f, Loader=yaml.FullLoader) or dict()
        page_id = cls._page_id(url)
        assert (
            page_id in meta_yaml
        ), "Url has not been registered. url={}".format(url)
        dic = meta_yaml[page_id]
        return cls.from_dict(dic)

    @classmethod
    def from_dict(cls, dic):
        return cls(dic["date_time"], dic["url"], dic["n_data_records"])

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

    def to_dict(self):
        return {
            "url": self.url,
            "page_id": self.page_id,
            "date_time": self.date_time,
            "n_data_records": self.n_data_records,
        }

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

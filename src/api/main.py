import os
import tempfile
import time
import uuid
import datetime

import flask
import flask_apispec
import flask_cors
import logging
import marshmallow
import pathlib
import urllib
import urllib.request
import urllib.error
import urllib.parse
import webargs
import yaml


import src.core as core
import src.utils as utils


app = flask.Flask(__name__)
cors = flask_cors.CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d : "
    "name=%(name)s : "
    "level=%(levelname)s : "
    "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


this_scripts_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(this_scripts_dir, "..", ".."))

# create output directory
outputs_dir = os.path.join(project_dir, "outputs")
outputs_dir = pathlib.Path(outputs_dir)
outputs_dir.mkdir(parents=False, exist_ok=True)

raw_htmls_dir = os.path.join(outputs_dir, "raw_htmls")
raw_htmls_dir = pathlib.Path(raw_htmls_dir)
raw_htmls_dir.mkdir(parents=False, exist_ok=True)

preprocessed_htmls_dir = os.path.join(outputs_dir, "preprocessed_htmls")
preprocessed_htmls_dir = pathlib.Path(preprocessed_htmls_dir)
preprocessed_htmls_dir.mkdir(parents=False, exist_ok=True)

intermediate_results_dir = os.path.join(outputs_dir, "intermediate_results")
intermediate_results_dir = pathlib.Path(intermediate_results_dir)
intermediate_results_dir.mkdir(parents=False, exist_ok=True)

results_dir = os.path.join(outputs_dir, "results")
results_dir = pathlib.Path(results_dir)
results_dir.mkdir(parents=False, exist_ok=True)

# create page's metadata file
pages_meta = os.path.join(outputs_dir, "pages-meta.yml")
pages_meta = pathlib.Path(pages_meta)
pages_meta.touch(exist_ok=True)


# todo replace this
def get_page_prefix(date_time: datetime.datetime, page_id: str):
    return "{date_time}-{page_id}-".format(
        date_time=date_time.strftime("%Y-%m-%d-%H%M"), page_id=page_id
    )


# todo replace this
def get_page_id(url):
    number = "{:+020d}".format(hash(url))[-9:]
    return "-".join((number[:3], number[3:6], number[6:9]))


class CallMdrSchema(marshmallow.Schema):
    class Meta:
        fields = ("output-filepath",)


@app.route("/api/", methods=["POST"])
@flask_cors.cross_origin()
@flask_apispec.use_kwargs({"url": webargs.fields.Url()})
@flask_apispec.marshal_with(CallMdrSchema)
def call_mdr(url):
    """todo(unittest)"""
    logging.info("Request to %s for url='%s'", call_mdr.__name__, url)

    start = time.time()
    output_filepath = execute(url)
    end = time.time()
    exec_time = end - start

    logging.info(
        "Finished successfully in %.3f sec. Output file_path='%s'",
        exec_time,
        output_filepath,
    )

    # todo(improvement) cache

    return {
        "output-filepath": output_filepath,
    }


class PageMeta(object):
    @staticmethod
    def _page_id(url):
        number = "{:+09x}".format(hash(url))[-9:]
        return "-".join((number[:3], number[3:6], number[6:9]))

    @staticmethod
    def is_registered(url):
        page_id = get_page_id(url)
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


@app.route("/api/save_page", methods=["POST"])
@flask_cors.cross_origin()
@flask_apispec.use_kwargs(
    {
        "url": webargs.fields.Url(required=True),
        "n_data_records": webargs.fields.Int(required=True),
    }
)
def save_page(url, n_data_records):
    """todo(unittest)"""
    logging.info("Request to save page: url='%s'", url)

    if PageMeta.is_registered(url):
        logging.info(
            "This page is already save. Returning without changing anything."
        )
        return

    meta = PageMeta.register(url, n_data_records)

    # raw_html_file = raw_htmls_dir.joinpath(page_files_prefix + "raw.html").absolute()
    #
    # logging.info("Downloading the html page.")
    # response = urllib.request.urlopen(url)
    # page = response.read()
    # with html_file:
    #     html_file.write(page)
    # logging.info("Done")

    logging.info("Saved successfully. page_id={}".format(meta.page_id))


def execute(url: str) -> str:
    """
        todo(unittest)
        Instantiate an MDR, call it and output the result in a file.
    Returns:
        The file path to the result file with a table.
    """

    now = datetime.datetime.now()
    page_id = get_page_id(url)
    prefix = get_page_prefix(now, page_id)
    html_file = tempfile.NamedTemporaryFile(
        dir=outputs_dir, delete=False, mode="wb", prefix=prefix, suffix=".html"
    )

    logging.info("Downloading the html page.")
    response = urllib.request.urlopen(url)
    page = response.read()
    with html_file:
        html_file.write(page)
    logging.info("Done")

    doc = utils.open_html_document(filepath=html_file.name)
    # doc = doc.body
    import lxml
    import lxml.etree

    lxml.etree.strip_elements(doc, "script")
    lxml.etree.strip_elements(doc, "style")

    import pdb

    # pdb.set_trace()

    mdr = core.MDR()
    logging.info("Processing MDR.")
    data_records = mdr(doc)
    logging.info("Done.")
    # try:
    #     pass
    # except:
    #     pdb.set_trace()
    n_data_records = len(data_records)

    logging.info("Found %d data records.", n_data_records)

    utils.paint_data_records(mdr, doc)
    colored_doc_str = lxml.etree.tostring(doc)

    prefix_colored = prefix + "colored-"
    colored_html_file = tempfile.NamedTemporaryFile(
        dir=outputs_dir,
        delete=False,
        mode="wb",
        prefix=prefix_colored,
        suffix=".html",
    )
    with colored_html_file:
        colored_html_file.write(colored_doc_str)

    prefix_data_records = prefix + "data-records-"
    data_records_file = tempfile.NamedTemporaryFile(
        dir=outputs_dir,
        delete=False,
        mode="w",
        prefix=prefix_data_records,
        suffix=".txt",
    )
    with data_records_file:
        import pprint

        data_records_file.write(
            core.MDR.DEBUG_FORMATTER.pformat(
                mdr.get_data_records_as_lists(node_as_node_name=True)
            )
        )

    # todo make util that gives the nodes names

    # output_filepath = os.path.join(outputs_dir, html_file.name)
    return colored_html_file.name

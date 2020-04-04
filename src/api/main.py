import time

import flask
import flask_apispec
import flask_cors
import logging

import lxml
import lxml.html
import lxml.etree
import marshmallow
import urllib
import urllib.request
import urllib.error
import urllib.parse
import webargs

import src.core as core
from src.files_management import PageMeta

app = flask.Flask(__name__)
cors = flask_cors.CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d : " "name=%(name)s : " "level=%(levelname)s : " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        "Finished successfully in %.3f sec. Output file_path='%s'", exec_time, output_filepath,
    )

    # todo(improvement) cache

    return {
        "output-filepath": output_filepath,
    }


@app.route("/api/save_page", methods=["POST"])
@flask_cors.cross_origin()
@flask_apispec.use_kwargs(
    {"url": webargs.fields.Url(required=True), "n_data_records": webargs.fields.Int(required=True)}
)
def save_page(url, n_data_records):
    """todo(unittest)"""
    logging.info("Request to save page: url='%s'", url)
    meta = save_page_execute(n_data_records, url, True)
    logging.info("Finished request successfully. page_id={}".format(meta.page_id))


def save_page_execute(n_data_records, url, download) -> PageMeta:
    is_already_registered = PageMeta.is_registered(url)
    meta = (
        PageMeta.register(url, n_data_records)
        if not is_already_registered
        else PageMeta.from_meta_file_by_url(url)
    )
    if is_already_registered:
        logging.info("This page is already registered. page_id={}".format(meta.page_id))
    else:
        logging.info("New page registered. page_id={}".format(meta.page_id))
    if download:
        logging.info("Downloading the html page. page_id={}".format(meta.page_id))
        try:
            response = urllib.request.urlopen(meta.url)
            page = response.read()
        except Exception as ex:
            logging.error(
                "Something went wrong during the download. page_id={} ex={}".format(
                    meta.page_id, ex
                )
            )
        else:
            with meta.raw_html.open(mode="wb") as f:
                f.write(page)
            logging.info("Done")
    return meta


def execute(url: str) -> str:
    """
        todo(unittest)
        todo remake this function
        Instantiate an MDR, call it and output the result in a file.
    Returns:
        The file path to the result file with a table.
    """
    page_meta = (
        PageMeta.register(url, None)
        if not PageMeta.is_registered(url)
        else PageMeta.from_meta_file_by_url(url)
    )

    import prepostprocessing as ppp

    ppp.download_raw(page_meta)
    ppp.cleanup_html(page_meta)
    doc = page_meta.get_preprocessed_html_tree()
    precomputed_distances = page_meta.load_precomputed_distances()
    logging.info(
        "Precomputed distances is full: %s", "yes" if len(precomputed_distances) > 0 else "no"
    )

    mdr = core.MDR.with_defaults(doc, precomputed_distances)
    logging.info("Processing MDR.")
    data_records = mdr()
    logging.info("Done.")
    # todo(improvement) save computed stuff

    n_data_records = len(data_records)
    logging.info("Found %d data records.", n_data_records)

    # todo revive this code
    import utils

    # core.paint_data_records(drecords)

    PageMeta.persist_html(page_meta.colored_html, doc)

    # with data_records_file:
    #     import pprint
    #
    #     data_records_file.write(
    #         core.MDR.DEBUG_FORMATTER.pformat(
    #             mdr.get_data_records_as_lists(node_as_node_name=True)
    #         )
    #     )

    # todo make util that gives the nodes names

    # output_filepath = os.path.join(outputs_dir, html_file.name)
    return str(page_meta.colored_html)

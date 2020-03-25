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
    format="%(asctime)s.%(msecs)03d : "
    "name=%(name)s : "
    "level=%(levelname)s : "
    "%(message)s",
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

    return "unavailable"

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
    meta = save_page_execute(n_data_records, url, True)
    logging.info(
        "Finished request successfully. page_id={}".format(meta.page_id)
    )


def save_page_execute(n_data_records, url, download) -> PageMeta:
    is_already_registered = PageMeta.is_registered(url)
    meta = (
        PageMeta.register(url, n_data_records)
        if not is_already_registered
        else PageMeta.from_meta_file(url)
    )
    if is_already_registered:
        logging.info(
            "This page is already registered. page_id={}".format(meta.page_id)
        )
    else:
        logging.info("New page registered. page_id={}".format(meta.page_id))
    if download:
        logging.info(
            "Downloading the html page. page_id={}".format(meta.page_id)
        )
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
    page_meta = PageMeta.register(url, None)

    logging.info("Downloading the html page.")
    response = urllib.request.urlopen(url)
    page = response.read()
    with page_meta.raw_html.open(mode="wb") as f:
        f.write(page)
    logging.info("Done")

    # doc = src.files_management.open_html_document(filepath=str(page_meta.raw_html))
    doc = page_meta.get_raw_html_tree(remove_stuff=True)
    lxml.etree.strip_elements(doc, "script")
    lxml.etree.strip_elements(doc, "style")

    mdr = core.MDR()
    logging.info("Processing MDR.")
    data_records = mdr(doc, page_meta)
    logging.info("Done.")

    n_data_records = len(data_records)
    logging.info("Found %d data records.", n_data_records)

    # todo revive this code
    # utils.paint_data_records(mdr, doc)
    # colored_doc_str = lxml.etree.tostring(doc)

    # prefix_colored = page_meta.prefix + "colored-"
    # colored_html_file = tempfile.NamedTemporaryFile(
    #     dir=outputs_dir,
    #     delete=False,
    #     mode="wb",
    #     prefix=prefix_colored,
    #     suffix=".html",
    # )
    # with colored_html_file:
    #     colored_html_file.write(colored_doc_str)
    #
    # prefix_data_records = page_meta.prefix + "data-records-"
    # data_records_file = tempfile.NamedTemporaryFile(
    #     dir=outputs_dir,
    #     delete=False,
    #     mode="w",
    #     prefix=prefix_data_records,
    #     suffix=".txt",
    # )
    # with data_records_file:
    #     import pprint
    #
    #     data_records_file.write(
    #         core.MDR.DEBUG_FORMATTER.pformat(
    #             mdr.get_data_records_as_lists(node_as_node_name=True)
    #         )
    #     )
    #
    # # todo make util that gives the nodes names
    #
    # # output_filepath = os.path.join(outputs_dir, html_file.name)
    # return colored_html_file.name

import os
import sys
import tempfile
import time
import uuid

import flask
import flask_apispec
import flask_cors
import logging
import marshmallow
import pathlib
import webargs

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
    "module=%(module)s : "
    "func=%(funcName)s : "
    "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


this_scripts_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(this_scripts_dir, "..", ".."))
outputs_dir = os.path.join(project_dir, "outputs")
pathlib.Path(outputs_dir).mkdir(parents=False, exist_ok=True)
logging.info("Outputs will be in the following folder: %s", outputs_dir)


# todo remove me
# HTMLS_DIR = os.path.join(OUTPUT_DIR, "htmls")
# pathlib.Path(HTMLS_DIR).mkdir(parents=False, exist_ok=True)


class CallMdrSchema(marshmallow.Schema):
    class Meta:
        fields = ("output-filepath",)


@app.route("/api/", methods=["POST"])
@flask_cors.cross_origin()
@flask_apispec.use_kwargs({"url": webargs.fields.Url()})
@flask_apispec.marshal_with(CallMdrSchema)
def call_mdr(url):
    """todo(unittest)"""
    request_id = uuid.uuid4().hex

    logging.info("Request to %s for url='%s'", call_mdr.__name__, url)

    start = time.time()
    output_filepath = execute(request_id, url)
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

    logging.info("Saved successfully")


def execute(request_id: str, url: str) -> str:
    """
        todo(unittest)
        Instantiate an MDR, call it and output the result in a file.
    Args:
        request_id:
        url:
    Returns:
        The file path to the result file with a table.
    """
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    prefix = "{now}-{request_id}-".format(now=now, request_id=request_id[-6:])
    html_file = tempfile.NamedTemporaryFile(
        dir=outputs_dir, delete=False, mode="wb", prefix=prefix, suffix=".html"
    )

    import urllib
    import urllib.request
    import urllib.error
    import urllib.parse

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

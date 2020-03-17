import os
import tempfile
import time
import uuid
from typing import Callable

import flask
import flask_apispec
import flask_cors
import logging
import marshmallow
import pathlib
import webargs


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


def _request_uuid_adder(u: str) -> Callable[[str], str]:
    def _adder(msg: str) -> str:
        return "request_uuid={0} : {1}".format(u, msg)

    return _adder


OUTPUT_DIR = ".pymdr/"
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), OUTPUT_DIR)
pathlib.Path(OUTPUT_DIR).mkdir(parents=False, exist_ok=True)


class CallMdrSchema(marshmallow.Schema):
    class Meta:
        fields = ("output-filepath",)


@app.route("/api/", methods=["POST"])
@flask_cors.cross_origin()
@flask_apispec.use_kwargs({"url": webargs.fields.Url()})
@flask_apispec.marshal_with(CallMdrSchema)
def call_mdr(url):
    """todo(unittest)"""
    u = uuid.uuid4().hex
    msg_with_uuid = _request_uuid_adder(u)

    logging.info(
        msg_with_uuid("Request to %s for url='%s'"), call_mdr.__name__, url
    )

    start = time.time()
    output_filepath = execute(url)
    end = time.time()
    exec_time = end - start

    logging.info(
        msg_with_uuid(
            "Finished successfully in %.3f sec. Output file_path='%s'"
        ),
        exec_time,
        output_filepath,
    )

    return {
        "output-filepath": output_filepath,
    }


def execute(url: str) -> str:
    """
        todo(unittest)
        Instantiate an MDR, call it and output the result in a file.
    Args:
        url:
    Returns:
        The file path to the result file with a table.
    """
    fp = tempfile.NamedTemporaryFile(
        dir=OUTPUT_DIR, delete=False, mode="w", suffix=".txt"
    )
    with fp:
        fp.write(url + "\n")
    output_filepath = os.path.join(OUTPUT_DIR, fp.name)
    return output_filepath

import os
import tempfile
import time

from pathlib import Path
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), ".pymdr/")
Path(OUTPUT_DIR).mkdir(parents=False, exist_ok=True)

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/api/", methods=["POST"])
@cross_origin()
def hello_world():
    url = request.json["url"]
    fp = tempfile.NamedTemporaryFile(
        dir=OUTPUT_DIR, delete=False, mode="w", suffix=".txt"
    )
    with fp:
        fp.write(url + "\n")
    output_filepath = os.path.join(OUTPUT_DIR, fp.name)
    time.sleep(1)
    return {
        "output-filepath": output_filepath,
    }

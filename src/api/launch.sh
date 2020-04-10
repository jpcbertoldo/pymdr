#!/usr/bin/env bash

# this file's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# start main.py adding src to the python path
PYTHONPATH="${DIR}/../../" FLASK_ENV=development FLASK_APP=${DIR}/main.py flask run -p 2020
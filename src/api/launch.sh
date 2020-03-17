#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FLASK_ENV=development FLASK_APP=${DIR}/main.py flask run -p 2020
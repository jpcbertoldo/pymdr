# pymdr
Python implementation of Mining Data Records.

**Reference paper**

Liu, B., Grossman, R., & Zhai, Y. (2003). Mining data records in web pages. InProceedings of theninth acm sigkdd international conference on knowledge discovery and data mining(p. 601–606). New York, NY, USA: Association for Computing Machinery.

# Installation

The `setup.py` has not been tested yet and is not safe to use. Please follow the instructions bellow.

Python Version 3.6.9

## Instructions

- Open a terminal **in the root directory** of the project

- Make sure you are using the right python version is 3.6.9

```bash
python3 -V
```

- Make sure you have `virtualenv` installed

```bash
pip install virtualenv==20.0.18
```

- Create a virtual environment and install the requirements (replace `apt` if you are not on ubuntu)

```bash
virtualenv venv -p python3.6
source ./venv/bin/activate
pip install -r requrirements/dev.txt
```

- Install `graphviz`

```bash
sudo apt install graphviz
```

- Add the src module to the PYTHONPATH in the virtualenv

```bash
PTH_FILE="$(pwd)/venv/lib/python3.6/site-packages/src.pth"
touch PTH_FILE
echo "$(pwd)/src" >> PTH_FILE
deactivate
source ./venv/bin/activate
```

- cleanup the nbs
- make outputs results available somehow
- write a good readme.md

# Next

Give a try to solving problems by using the cleanup_all from the NodeNamer in the dist computation?
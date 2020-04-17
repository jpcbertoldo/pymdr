from setuptools import setup, find_packages

setup(
    name="pymdr",
    version="0.1",
    packages=find_packages(include=("src",), exclude="extension"),
    author="Joao P C Bertoldo",
    author_email="joaopcbertoldo@gmail.com",
    description="An open source python implementation of the algorithm MDR proposed by Liu, Bing et al. (2003) in "
    "'Mining data records in Web pages'.",
    url="https://github.com/joaopcbertoldo/pymdr",
    project_urls={"Project Report": "https://github.com/joaopcbertoldo/pymdr-project-report"},
)

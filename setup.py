import os
import pathlib

import pkg_resources
from setuptools import find_packages, setup

# Basic Information
NAME = "inpaint"
VERSION = "0.1.0"
DESCRIPTION = "A python package for image inpainting"
REPOSITORY = "https://github.com/prajnan93/image-inpainting"

# Define the classifiers
# See https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "Programming Language :: Python :: 3.8",
]

# Define the keywords
KEYWORDS = [
    "pytorch",
    "machine learning",
    "deep learning",
    "image inpainting",
    "computer vision",
]

# Directories to ignore in find_packages
EXCLUDES = ()

# Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"


with pathlib.Path("requirements.txt").open() as requirements_txt:
    INSTALL_REQUIRES = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]


CONFIG = {
    "name": NAME,
    "version": VERSION,
    "description": DESCRIPTION,
    "classifiers": CLASSIFIERS,
    "keywords": KEYWORDS,
    "url": REPOSITORY,
    "packages": find_packages(
        where=PROJECT, include=["inpaint", "inpaint.*"], exclude=EXCLUDES
    ),
    "install_requires": INSTALL_REQUIRES,
    "python_requires": ">=3.6",
    "test_suite": "tests",
    "tests_require": ["pytest>=3"],
    "include_package_data": True,
}

if __name__ == "__main__":
    setup(**CONFIG)

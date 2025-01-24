#!/usr/bin/env python

from setuptools import find_packages, setup

VERSION = "0.6"

with open("README.md", encoding="UTF-8") as f:
    readme = f.read()

with open("requirements.txt", encoding="UTF-8") as f:
    required = f.read().splitlines()

setup(
    name="nucleus",
    version=VERSION,
    description="Nucleus detects individual nuclei in crowded tissue-like confocal microscopy images",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=['segmentation','2D','cryosections','mask R-CNN','Detectron2'],
    author="Tiago Rito",
    author_email="tiago.rito@crick.ac.uk",
    url="https://github.com/tiagu/Nucleus",
    license="MIT",
    python_requires=">=3.7",
    install_requires=required,
    packages=find_packages(exclude="notebooks"),
    include_package_data=True,
    zip_safe=False,
)

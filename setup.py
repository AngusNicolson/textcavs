from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Generate textual explanations for a target NN"

setup(
    name="textcavs",
    version=VERSION,
    author="Angus Nicolson",
    author_email="<angusjnicolson@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages()
)

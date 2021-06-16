"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from pathlib import Path

curr_dir = Path(__file__).parent

with open(curr_dir/"requirements.txt", "r") as f:
    # include all requirements which are not commented
    install_requires = [i for i in f.read().splitlines() if i[0] != "#"]

description = r"""Named after a spell in the Harry Potter Universe, where it 
    amplies the sound of a speaker. In muggles' terminology, this is a repository 
    of modules for audio and speech processing for and on top of machine learning 
    based tasks such as speech-to-text."""

README = (curr_dir/"README.md").read_text()

setup(name="sonorus",
    version="0.1.0",
    description=description,
    long_descriptio=README,
    author="Md Imbesat Hassan Rizvi",
    author_email="imbugene@gmail.com",
    url="https://github.com/pensieves/sonorus",
    packages=find_packages(exclude=["contrib", "docs", "examples", "tests"]), # Required
    install_requires=install_requires,
    license="MIT",
    keywords=["deep learning", "speech recognition", "speech to text", "language modelling"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Langauge :: Python :: 3",
    ],
)

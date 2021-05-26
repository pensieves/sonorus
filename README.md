# sonorus
Named after a spell in the Harry Potter Universe, where it amplies the sound of a speaker. In muggles' terminology, this is a repository of modules for audio and speech processing for and on top of machine learning based tasks such as speech-to-text.

## Getting Started:

### Installation:
*Install dependencies*

The repository has dependencies such as `kenlm`, `pyflashlight`, `fairseq` and `portaudio` which needs to be installed before pip-installable modules

To install `kenlm` with python bindings, refer to the `kenlm` [github repository](https://github.com/kpu/kenlm).

To install `pyflashlight` with python bindings, refer to the [installation instructions](https://github.com/flashlight/flashlight/tree/master/bindings/python#installation). NOTE that the C++ build itself is not necessarily required for building python bindings. FURTHERMORE, `pyflashlight` will soon be made `pip`-installable via `pypi`.

To install `fairseq`, refer to [requirements and installations](https://github.com/pytorch/fairseq) from the `fairseq` github repository. NOTE that the current `pip`-installable `pypi` module is of version < 1.0 and hence installation from source is currently required. Once the `pypi` index is updated with the latest `fairseq` package, the same can be installed using `pip`.

`pyaudio` has a dependency on `portaudio`. If not using conda, make sure `portaudio` is installed. For example, for Ubuntu, the same can be installed by executing:

`sudo apt install portaudio19-dev`

Finally, install requirements by executing:

`pip install -r requirements.txt`

or install using conda in a conda environment.

### Environment set up:

*Note:* Environment set up is required while using Google Cloud's speech to text api. For this, Google Application Credentials is to be set as an environment variable by exporting e.g.: 
```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-cloud-credentials.json
```

### Sample running instructions:

- Receives speech input from microphone and prints it on console using on-device Facebook's Wav2Vec2 model made available by Hugging Face..

`python3 run/speech.py`

To modify the execution parameters of the on-device model such as providing GPU device index in case of availability, the program can be run as:

`python3 run/speech.py --gpu_idx 0`

- For using Google cloud's speech to text execute:

`python3 run/google-speech.py`

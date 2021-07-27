import wget
import hashlib
from pathlib import Path
import shutil
from copy import deepcopy

from sonorus import CACHE_DIR
from sonorus.utilities.utils import unpack_archive

LIBRISPEECH_TGSMALL_URL = "https://www.dropbox.com/s/eecyok5h1pfn7vy/librispeech_tgsmall.tar.gz?dl=1"

MODEL_ITEM_FILENAMES = dict(
    model_rxfilename="final.mdl",
    graph_rxfilename="HCLG.fst",
    symbols_filename="words.txt",
    tree_rxfilename="tree",
    lexicon_rxfilename="L.fst",
    disambig_rxfilename="disambig.int",
    phoneme_file="phones.txt",
)


def download_model(
    url=LIBRISPEECH_TGSMALL_URL, 
    model_item_filenames=MODEL_ITEM_FILENAMES,
    cache_dir=CACHE_DIR,
    force_download=False,
):
    
    url_hash = hashlib.md5(url.encode()).hexdigest()
    download_dir = Path(cache_dir)/f"kaldi_model_{url_hash}"
    model_dir = check_and_download(url, download_dir, force_download=force_download)

    model_item_filepaths = deepcopy(model_item_filenames)
    for item, filename in model_item_filepaths.items():
        # glob object is a generator hence using next to get first result
        model_item_filepaths[item] = next(model_dir.glob(f"**/{filename}"))

    return model_dir, model_item_filepaths


def check_and_download(url, download_dir, force_download=False, retry=5):

    if force_download and download_dir.exists():
        shutil.rmtree(download_dir, ignore_errors=True)

    url_file = download_dir/"url.txt"
    
    if not download_dir.exists():

        download_dir.mkdir(parents=True)

        print(f"Downloading model from {url} ...")
        for i in range(retry):
            
            try:
                print(f"Attempting download {i+1}/{retry} times ...")
                archive_filename = wget.download(url, out=str(download_dir))
                print("Download completed")
                break
            
            except Exception as e:
                if i < retry-1:
                    print(f"Exception occured: {e}")
                continue
        
        model_dir = unpack_archive(archive_filename, unpack_dir=download_dir)

        with open(url_file, "w") as f:
            f.write(f"{model_dir}\t{url}")

        Path(archive_filename).unlink(missing_ok=True)

    else:
        print(f"Model from {url} already downloaded, cached model will be used.")
        with open(url_file, "r") as f:
            model_dir = Path(f.read().split("\t")[0])

    return model_dir
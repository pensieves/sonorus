set -e

MKL_SCRIPT="$PWD/install_mkl.sh"
MKL_SCRIPT_URL="https://github.com/eddelbuettel/mkl4deb/raw/master/script.sh"
FLASHLIGHT_GIT="https://github.com/flashlight/flashlight.git"
FLASHLIGHT_DIR="$PWD/flashlight"

if [ "$INSTALL_MKL" == "1" ]; then
    wget -O $MKL_SCRIPT $MKL_SCRIPT_URL
    chmod +x $MKL_SCRIPT
    $MKL_SCRIPT
    rm $MKL_SCRIPT
    echo "MKL installed!"
else  
    echo "Instructed to omit MKL installation."\
        "Ensure MKL is installed or \"export INSTALL_MKL=1\""
fi

# install system libraries
apt-get install -y \
    libatlas-base-dev \
    libfftw3-dev

# install pip dependencies
pip install packaging

# install flashlight
if [ -d "$FLASHLIGHT_DIR" ]; then
    rm -rf $FLASHLIGHT_DIR
fi

echo "Installing pyflashlight ..."
git clone $FLASHLIGHT_GIT $FLASHLIGHT_DIR
cd $FLASHLIGHT_DIR/bindings/python
python3 setup.py install
pip3 install -e .

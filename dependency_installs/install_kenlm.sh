set -e

KENLM_GIT="https://github.com/kpu/kenlm.git"
KENLM_PIP="https://github.com/kpu/kenlm/archive/master.zip"
KENLM_DIR="$PWD/kenlm"

# install system libraries
apt-get install -y \
    libeigen3-dev \
    libboost-all-dev \
    zlibc \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev

# install kenlm
if [ -d "$KENLM_DIR" ]; then
    rm -rf $KENLM_DIR
fi

echo "Installing kenlm ..."
git clone $KENLM_GIT $KENLM_DIR
cd $KENLM_DIR
mkdir -p build
cd build
cmake ..
make -j 4
echo "export KENLM_ROOT=$KENLM_DIR" >> ~/.bashrc
source ~/.bashrc
echo "kenlm installed!"

# install python bindings of kenlm
echo "Installing python bindings of kenlm ..."
pip install $KENLM_PIP
echo "python bindings of kenlm installed!"
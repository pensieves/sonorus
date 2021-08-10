set -e

DIR=$(dirname "$(readlink -f "$0")")

chmod +x $DIR/install_kenlm.sh
chmod +x $DIR/install_pyflashlight.sh
chmod +x $DIR/install_fairseq.sh

$DIR/install_kenlm.sh

export USE_CUDA="0" # set to 1 if CUDA is installed and path is accessible
export INSTALL_MKL="1" # set to 0 if already installed
export KENLM_ROOT="$PWD/kenlm"

$DIR/install_pyflashlight.sh
$DIR/install_fairseq.sh
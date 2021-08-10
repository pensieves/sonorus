set -e

FAIRSEQ_GIT="https://github.com/pytorch/fairseq.git"
FAIRSEQ_DIR="$PWD/fairseq"

git clone $FAIRSEQ_GIT $FAIRSEQ_DIR
cd $FAIRSEQ_DIR
pip install --editable ./

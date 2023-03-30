
#
# Preprocess data. Takes around 1.5 hours.
#

echo Starting ...;   date

python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16

# mkdir -p moses-processed
# mv tensor* moses-processed

echo Completed preprocess; date


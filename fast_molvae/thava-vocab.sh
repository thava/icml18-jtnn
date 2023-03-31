
#
# Generate vocabulary for your own dataset.
# Takes around 1.5 hours to complete. Single Threaded.
#
# See fast_molvae/README.md for more information.
#

python ../fast_jtnn/mol_tree.py < ../data/moses/train.txt | tee ../data/moses/vocab-all-$$.txt



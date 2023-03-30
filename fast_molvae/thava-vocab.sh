
#
# Generate vocabulary ...
# Takes around 1.5 hour. Single Threaded.
#

python ../fast_jtnn/mol_tree.py < ../data/moses/train.txt | tee ../data/moses/vocab-all-$$.txt



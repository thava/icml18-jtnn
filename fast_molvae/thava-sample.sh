
#
# Sample new molecules with trained models.
# See fast_molvae/README.md for more information.
#

python sample.py --nsample 30000 --vocab ../data/moses/vocab.txt --hidden 450 --model moses-h450z56/model.iter-700000 > mol_samples.txt

# python sample.py --nsample 100 --vocab ../data/moses/my-vocab.txt --hidden 450 --model vae_model/model.iter-200000  > mol_samples.txt


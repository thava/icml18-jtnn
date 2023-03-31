
#
# Train VAE model with KL annealing
# See fast_molvae/README.md for more information.
#

date
echo Starting Training

mkdir -p vae_model/
python vae_train.py --train moses-processed --vocab ../data/moses/vocab.txt --save_dir vae_model/

date
echo Finished Training

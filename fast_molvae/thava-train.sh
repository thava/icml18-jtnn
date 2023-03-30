
# python vae_train.py --train moses-processed --vocab ../data/moses/vocab.txt --save_dir vae_model/

echo Starting Training
date

mkdir -p vae_model/
python vae_train.py --train moses-processed --vocab ../data/moses/vocab-thava.txt --save_dir vae_model/

echo Finished Training
date

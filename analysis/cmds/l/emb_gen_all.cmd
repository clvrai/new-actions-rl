# Must manually specify emb model file like: 
# --load-emb-model-file /home/ayush/icml/data/embedder/trained_models/create_g1-htvae-5000.m 
# Ran on lim-b 
python main.py --env-name CreateLevelPush-v0 --play-env-name StateCreateGamePlay-v0  --save-embeddings-file create_all --prefix debug --log-dir ~/tmp/gym --split-type all_clean --play-data-folder /home/ayush/data/create_gran1_len7 --create-play-len 7 --load-emb-model-file  create_all_len7-htvae-10000.m

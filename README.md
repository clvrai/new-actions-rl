# Generalization to New Actions in Reinforcement Learning
[Ayush Jain](http://www-scf.usc.edu/~ayushj/)\*, [Andrew Szot](https://www.andrewszot.com)\*, [Joseph J. Lim](https://clvrai.com) at [USC CLVR lab](https://clvrai.com)  
[[Paper website](https://sites.google.com/view/action-generalization)] 

## Directories
The structure of the repository:  
  - `analysis`: Scripts used for analysis figures and experiments.  
  - `envs`: the four subfolders in this folder contain the four environments.   
  - `method`: Implementation of all method and baseline details   
  - `rlf`: Reinforcement Learning Framework. General RL / PPO training code.   
  - `scripts`: Miscalaneous scripts. Contains script for generating the train /
    test action set splits.  
  - `main.py`: Entry point for running policy.  
  - `embedder.py`: Entry point for training embedder.  


Log directories:  

  - `data/trained_model/ENV-NAME_PREFIX/`: Trained models.  
  - `data/vids/ENV-NAME/`: Evaluation videos.  
  - `data/logs/ENV-NAME/PREFIX/`: Tensorboard summary.  


## Prerequisites
- Python 3.7
- MuJoCo 2.0

## Dependencies
All the python package requirements are in `requirements.txt`. If you are using conda, you can use the following command with Python 3.7.3:
```
conda create -n [your_name] python=3.7
source activate [your_name]
pip install -r requirements.txt
```

## Experiments
The experiment flow for each environment is similar. The steps are always the
same as follows:  

- Generate train and test action splits: `python gen_action_sets.py --env-name $ENV_NAME`

- Generate Action Datasets for the environment: `python embedder.py --env-name $PLAY_ENV_NAME --save-dataset`

- Train action embedder model: `python embedder.py --env-name $PLAY_ENV_NAME --save-emb-model-file $EMB_FILE_NAME --train-embeddings`

- Generate embedding files: `python main.py --env-name $ENV_NAME --play-env-name $PLAY_ENV_NAME --load-emb-model-file $EMB_MODEL_NAME --save-embeddings-file $EMB_FILE_NAME --prefix main`

- Train policy with saved embeddings: `python main.py --env-name $ENV_NAME --load-embeddings-file $EMB_FILE_NAME`

Note:  
(1) `$EMB_MODEL_NAME` must be `$EMB_FILE_NAME-htvae-500.m` if your model is trained for at least 500 epochs (specified by `--emb-epochs`).  
(2) Use `--n-trajectories 64` and `--emb-epochs 500` for faster data generation and embedder training.

Below are the example commands used for each environment and method approach.

## Environments
### CREATE (Chain Reaction Tool Environment)

`$ENV_NAME` = `'CreateLevelPush-v0'` or `'CreateLevelNavigate-v0'` or `'CreateLevelObstacle-v0'`.  
`$PLAY_ENV_NAME` = `'StateCreateGameN1PlayNew-v0'` (state-based) or `'CreateGamePlay-v0'` (video-based).  
`$EMB_FILE_NAME` = `'create_st'` (state-based) or `create_im` (video-based)

(1) Train policy directly with:

`python main.py --env-name CreateLevelPush-v0 --prefix main`.  
`python main.py --env-name CreateLevelNavigate-v0 --prefix main`.  
`python main.py --env-name CreateLevelObstacle-v0 --prefix main`.  

OR

(2) For full procedure, follow these commands:

- Generate Splits: `python gen_action_sets.py --env-name CreateLevelPush-v0`
- Generate Data: `python embedder.py --env-name StateCreateGameN1PlayNew-v0 --save-dataset`
- Train Action Embedder: `python embedder.py --env-name StateCreateGameN1PlayNew-v0 --save-emb-model-file create_st --train-embeddings`
- Generate embedding files: `python main.py --env-name CreateLevelPush-v0 --play-env-name StateCreateGameN1PlayNew-v0 --load-emb-model-file create_st-htvae-5000.m --save-embeddings-file create_st --prefix main`
- Train policy with saved embeddings: `python main.py --env-name CreateLevelPush-v0 --load-embeddings-file create_st --prefix main`


### Reco 
There is no data generation or embedding learning to recommender system

`$ENV_NAME` = `'RecoEnv-v0'`

(1) Train policy directly with:

`python main.py --env-name RecoEnv-v0 --prefix main`

OR

(2) For full procedure, follow these commands:

- Generate Splits: `python gen_action_sets.py --env-name RecoEnv-v0`
- Policy: `python main.py --env-name RecoEnv-v0 --prefix main`


### Block Stacking

`$ENV_NAME` = `'StackEnv-v0'`   
`$PLAY_ENV_NAME` = `'BlockPlayImg-v0'`  
`$EMB_FILE_NAME` = `'stack_im'`

(1) Train policy directly with:

`python main.py --env-name StackEnv-v0 --prefix main`

OR

(2) For full procedure, follow these commands:

- Generate Splits: `python gen_action_sets.py --env-name StackEnv-v0`
- Generate Data: `python embedder.py --env-name BlockPlayImg-v0 --save-dataset`
- Train Action Embedder: `python embedder.py --env-name BlockPlayImg-v0 --save-emb-model-file stack_im --train-embeddings`
- Generate embedding files: `python main.py --env-name StackEnv-v0 --play-env-name BlockPlayImg-v0 --load-emb-model-file stack_im-htvae-5000.m --save-embeddings-file stack_im --prefix main`
- Train policy with saved embeddings: `python main.py --env-name StackEnv-v0 --load-embeddings-file stack_im --prefix main`

### Grid world

`$ENV_NAME` = `'MiniGrid-LavaCrossingS9N1-v0'`   
`$PLAY_ENV_NAME` = `'MiniGrid-Empty-Random-80x80-v0'`  
`$EMB_FILE_NAME` = `'gw_onehot_new'`


(1) Train policy directly with:

`python main.py --env-name MiniGrid-LavaCrossingS9N1-v0 --prefix main`

OR

(2) For full procedure, follow these commands:

- Generate Splits: `python gen_action_sets.py --env-name MiniGrid-LavaCrossingS9N1-v0`
- Generate Data: `python embedder.py --env-name MiniGrid-Empty-Random-80x80-v0 --save-dataset`
- Train Action Embedder: `python embedder.py --env-name MiniGrid-Empty-Random-80x80-v0 --save-emb-model-file gw_onehot_new --train-embeddings`
- Generate embedding files: `python main.py --env-name MiniGrid-LavaCrossingS9N1-v0 --play-env-name MiniGrid-Empty-Random-80x80-v0 --load-emb-model-file gw_onehot_new-htvae-5000.m --save-embeddings-file gw_onehot_new --prefix main`
- Train policy with saved embeddings: `python main.py --env-name MiniGrid-LavaCrossingS9N1-v0 --load-embeddings-file gw_onehot_new --prefix main`


## Baselines and Ablations

To run the baselines for any environment, add the following to the main command:

**Baselines**

- *Nearest-Neighbor (NN)*: `--nearest-neighbor --fixed-action-set --action-random-sample False --prefix NN`
- *Distance-based Policy Architecture (Dist)*: `--distance-based --prefix dist`
- *Non-hierarchical embeddings (VAE)*: `--load-embeddings-file $FILE --prefix vae`, where $FILE storing these embeddings is environment-dependent:
  - CREATE: `create_fc_st_vae`
  - Shape Stacking: `stack_vae`
  - Grid World: `gw_onehot_vae`

**Ablations**

- *Fixed Action Space (FX)*: `--fixed-action-set --action-random-sample False --prefix FX`
- *Random-Sampling without clustering (RS)*: `--sample-clusters False --prefix RS`
- *No-entropy (NE)*: `--entropy-coef 0. --prefix NE`

**Other embedding data formats**

- CREATE: *Video*-based embeddings: `--load-embeddings-file create_fc_im --o-dim 128 --z-dim 128 --prefix im`
- Grid World: (x,y) coordinate *state*-based embeddings: `--load-embeddings-file gw_st --prefix st`

**Ground-truth embeddings**

 for CREATE and Grid World: `--gt-embs --prefix GT`


## Analysis
For running the three analysis scripts simply run   
- `analysis/analysis_dist.py`.  
- `analysis/analysis_emb.py`.  
- `analysis/analysis_ratio.py`


### Acknowledgement

- PPO code is based on the [Pytorch implementation of PPO by Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
- The Grid world environment is from https://github.com/maximecb/gym-minigrid
- The recommender systems environment is from https://github.com/criteo-research/reco-gym

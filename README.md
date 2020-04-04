# Reinforcement Learning-based Black-box Adversarial Attacks on Gaussian-mixture-model Speech Anti-spoofing systems

This repository implemented the method that attacks anti-spoofing systems using reinforcement learning in black-box settings. Our attack targets are gaussian-mixture-model anti-spoofing classifiers provided by *ASVspoof2019* organizers. For the reinforcement learning algorithms, we used Deep Deterministic Policy Gradient (DDPG), and referred to [this repository](https://github.com/ikostrikov/pytorch-ddpg-naf).

### 1 Clone this repository and install dependencies.

Before installing packages, please prepare a new environment with **Python 3.6**. Because the GMMs need to use matlab engine in the python environment. Python-matlab engine is not supported in Python>3.6.

```shell
conda create -n [env name] python=3.6
```

Then please install the matlab engine. Firstly please go to the place where you installed matlab and find the directory for external engines. For example, it is as follows.

```shell
cd /software/MATLAB/R2018a/extern/engines/python
```

Then intall matlab engines using the following command.

```shell
python setup.py build --build-base=$(mktemp -d) install
```

Clone this repository and install packages.

```shell
git clone https://github.com/MingruiYuan/rlattack.git && pip install -r requirements.txt
```

The last thing is not to forget to overwrite the root directory for this repository on your device. Please look at the configuration file **config.json**. My example is as follows.

```json
"ROOT_DIR": "/scratch/myuan7/program/rlattack/"
```

### 2 Download datasets and data preprocessing.

We use the **logical access (LA)** part of **ASVspoof2019** dataset. Download it with the following command.

```shell
wget -bc https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
```

After unzipping the file, please also remember to change the path for your dataset. My example is as follows.

```json
"DATA_DIR": "/scratch/myuan7/corpus/LA/"
```

Preprocess the dataset. This will do the voice activity detection (VAD) and generate new protocols.

```shell
python -u main.py -S preprocess -C config.json
```

### 3 Training

In the process of training, we save checkpoints and evading audio utterances regularly. In order to distinguish between each run, we use starting time as the tag of each run. 

```shell
# ctime=$(date "+%y-%m-%d_%H-%M-%S")
# checkpoints are stored at ${ROOT_DIR}/saved_models/attack/${ctime}
# evading audio utterances (utterances that evade the detection of anti-spoofing systems) are stored at ${ROOT_DIR}/evading_audio/${ctime}
```

The following example will train the RL model with CQCC-PI-GMM anti-spoofing system from scratch.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py -S train_attack -C config.json -T ${ctime} --feature_type CQCC
```

Another example: Train the RL model with LFCC-LO-GMM from scratch.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py -S train_attack -C config.json -T ${ctime} --feature_type LFCC --label_only
```

Another example: Continue training from a checkpoint.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py -S train_attack -C config.json -T ${ctime} --feature_type CQCC --label_only --actor_path /path/to/actor.pt --critic_path /path/to/critic.pt
```

### 4 Evaluation

In the process of evaluation, we save evading audio utterances regularly. In order to distinguish between each run, we use starting time as the tag of each run. 

Evaluate with saved models.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py -S eval_attack -C config.json -T ${ctime} --feature_type CQCC - actor_path /path/to/actor.pt --critic_path /path/to/critic.pt
```

Evaluate with untrained models.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py -S eval_attack -C config.json -T ${ctime} --feature_type CQCC
```


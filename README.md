# Soft Actor-Critic in PyTorch
A PyTorch implementation of [Soft Actor-Critic](https://arxiv.org/pdf/1812.05905.pdf). I tested it with [Gym](https://gym.openai.com/) and [MuJoCo](http://www.mujoco.org/) environments (only which have continuous action spaces). 

## Requirements
- Clone this repository.
- python 3.6 is recommended.
- Use command `pip install -r requirements.txt` to install libraries.
- Install `mujoco_py` following instructions [here](https://github.com/openai/mujoco-py). (You need a license)

## Usage
To train the SAC agent, execute the command like this.
```
python code/main.py \
--env_name HalfCheetah-v2 \
--automatic_entropy_tuning True \
--cuda
```

To test the trained agent, execute the command like this.
```
python code/main.py \
--env_name HalfCheetah-v2 \
--log_dir ${trained log directory}
--cuda
```

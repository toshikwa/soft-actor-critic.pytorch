# Soft Actor-Critic in PyTorch
A PyTorch implementation of [Soft Actor-Critic](https://arxiv.org/pdf/1812.05905.pdf). I tested it with [Gym](https://gym.openai.com/) and [MuJoCo](http://www.mujoco.org/) environments (only which have continuous action spaces). I tried to make it fully commented. I wish it encourages you to understand algorithms and implementations well.

To do this, I really refered to some implementations.
- [OpenAI Spinning Up's implementation](https://github.com/openai/spinningup)
- [pranz24's implementation](https://github.com/pranz24/pytorch-soft-actor-critic)

## Requirements
- Clone this repository.
- python 3.6 is recommended.
- Use command `pip install -r requirements.txt` to install libraries.
- Install `mujoco_py` following instructions [here](https://github.com/openai/mujoco-py). (You need a license)

## Training
For training details, please refer to [the original paper](https://arxiv.org/pdf/1812.05905.pdf). I tested SAC agents only with learned temperature, which corresponds to blue lines of Figure 1 in the paper.

#### Hopper-v2

Train 1000000 steps with learned temperature.

```bash
python code/main.py --env_name Hopper-v2 \
--automatic_entropy_tuning True --num_steps 1000000 --cuda
```

#### Walker-v2

Train 3000000 steps with learned temperature.

```bash
python code/main.py --env_name Walker2d-v2 \
--automatic_entropy_tuning True --num_steps 3000000 --cuda
```

#### HalfCheetah-v2

Train 3000000 steps with learned temperature.

```bash
python code/main.py --env_name HalfCheetah-v2 \
--automatic_entropy_tuning True --num_steps 3000000 --cuda
```

#### Ant-v2

Train 3000000 steps with learned temperature.

```bash
python code/main.py --env_name Ant-v2 \
--automatic_entropy_tuning True --num_steps 3000000 --cuda
```

#### Humanoid-v2(MuJoCo)

```bash
python code/main.py --env_name Humanoid-v2 \
--automatic_entropy_tuning True --alpha 0.05 --num_steps 10000000 --cuda
```



## Results

Results will be like this. (I haven't tested all environments.) Note that I plotted results with standard deviations of 10 seeds, although Figure 1 in the paper is with min/max of 10 seeds.

![halfcheetah.png](https://github.com/ku2482/soft-actor-critic.pytorch/blob/result/results/halfcheetah.png)



## Test

Test the trained agent.

```bash
python code/main.py \
--env_name ${trained environment} \
--log_dir ${trained log directory} \
--cuda
```


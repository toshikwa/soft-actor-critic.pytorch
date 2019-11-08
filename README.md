# Soft Actor-Critic in PyTorch
A PyTorch implementation of Soft Actor-Critic[[1,2]](#references).
If you want to train a distributed version of Soft Actor-Critic, please reffer to [rltorch](https://github.com/ku2482/rltorch) repository.


## Requirements
You can install liblaries using `pip install -r requirements.txt` except `mujoco_py`.

Note that you need a licence to install `mujoco_py`. For installation, please follow instructions [here](https://github.com/openai/mujoco-py).

## Examples
You can train Soft Actor-Critic agent like this example [here](https://github.com/ku2482/soft-actor-critic.pytorch/blob/rltorch/code/main.py).

```
python code/main.py \
[--env_id str(default HalfCheetah-v2)] \
[--cuda (optional)] \
[--seed int(default 0)]
```

## Results

Results will be like this. (I haven't tested all environments.) Note that I plotted results with standard deviations of 10 seeds, although Figure 1 in the paper is with min/max of 10 seeds.

![halfcheetah](https://user-images.githubusercontent.com/37267851/68479002-9c4d6080-0274-11ea-9707-e6a297012428.png)

## References
[[1]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[2]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

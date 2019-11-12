# Soft Actor-Critic in PyTorch
A PyTorch implementation of Soft Actor-Critic[[1,2]](#references) with n-step rewards and prioritized experience replay[[3]](#references).

If you want to train a distributed version of Soft Actor-Critic, please reffer to [rltorch](https://github.com/ku2482/rltorch) repository.


## Requirements
You can install liblaries using `pip install -r requirements.txt` except `mujoco_py`.

Note that you need a licence to install `mujoco_py`. For installation, please follow instructions [here](https://github.com/openai/mujoco-py).

## Examples
You can train Soft Actor-Critic agent like this example [here](https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/main.py).

```
python code/main.py \
[--env_id str(default HalfCheetah-v2)] \
[--cuda (optional)] \
[--seed int(default 0)]
```

If you want to use n-step rewards and prioritized experience replay, set `multi_step=5` and `per=True` in configs.

## Results

Results of above example (without n-step rewards nor prioritized experience replay) will be like below, which are comparable (or better) with results of the paper.

<img src="https://user-images.githubusercontent.com/37267851/68636404-b752f580-053e-11ea-8e7a-9d631ab88868.png" title="cheetah" width=400><img src="https://user-images.githubusercontent.com/37267851/68636405-b7eb8c00-053e-11ea-9655-8bca41a3988f.png" title="walker" width=400>

<img src="https://user-images.githubusercontent.com/37267851/68636406-b7eb8c00-053e-11ea-8697-333844d2e500.png" title="ant" width=400><img src="https://user-images.githubusercontent.com/37267851/68636403-b752f580-053e-11ea-89cc-1f8759908f88.png" title="humanoid" width=400>


## References
[[1]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[2]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

[[3]](https://arxiv.org/abs/1511.05952) Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

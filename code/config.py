import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default="MountainCarContinuous-v0",
                        help='name of the environment to run')
    parser.add_argument('--tag', type=str, default="test",
                        help='name of the trial')
    parser.add_argument('-v', '--vis', action="store_true",
                        help='if render or not')
    parser.add_argument('--eval_per_steps', type=int, default=10000,
                        help='evaluate per steps')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for the reward')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='temperature parameter')
    parser.add_argument('--automatic_entropy_tuning', type=bool,
                        default=True, metavar='G',
                        help='if automaticaly adjust the temperature')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size')
    parser.add_argument('--num_steps', type=int, default=3000000, metavar='N',
                        help='maximum number of steps')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        metavar='N', help='model updates per simulator step')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='steps sampling random actions')
    parser.add_argument('--target_update_interval', type=int, default=1,
                        metavar='N', help='value target updates per steps')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer')
    parser.add_argument('--cuda', action="store_true",
                        help='if use gpu')
    parser.add_argument('--logdir', type=str, default="",
                        help='log directory which the model is saved.')
    parser.add_argument('--test', action="store_true",
                        help='if test')
    return parser.parse_args()

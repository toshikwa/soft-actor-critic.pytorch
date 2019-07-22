import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_return_history(
        mean_history, std_history, save_path, title, interval=10):
    # x axis
    x = np.arange(interval, len(mean_history)*interval+1, interval)
    tag = ['', ' (thousand)', ' (million)']
    i_tag = 0

    while interval >= 1000:
        x = x / 1000
        interval = interval / 1000
        i_tag += 1

    # plot
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.fill_between(
        x, mean_history - std_history, mean_history + std_history,
        alpha=0.3)
    plt.plot(x, mean_history, "-")
    plt.xlabel(f"steps{tag[i_tag]}")
    plt.ylabel(f"average return")
    plt.savefig(save_path)
    # plt.close()

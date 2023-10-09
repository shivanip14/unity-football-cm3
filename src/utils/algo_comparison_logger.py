from definitions import ROOT_DIR
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def plot_graph(stats_dict, y_label, algos):
    print("\nPlotting " + y_label)
    fig, ax = plt.subplots()
    axes = [ax, ax.twiny(), ax.twiny()]
    ax.set_ylabel(y_label)
    ax.set_xlabel("env_steps")
    fig.subplots_adjust(top=0.75)
    axes[0].set_frame_on(True)
    axes[0].patch.set_visible(False)
    inset_ax = inset_axes(ax, 2, 1, loc=7)
    inset_ax.tick_params(axis='x', which='both', bottom=False, top=False, labeltop=False, labelbottom=False)
    inset_ax.grid(axis='x')
    inset_ax.grid(axis='y')
    colors = ["#A0C75C", "#DA4C4C", "#3282F6"]
    for i, algo in enumerate(algos):
        axes[i].plot(stats_dict[algo]["x"], stats_dict[algo]["y"], label=algo, color=colors[i])
        if y_label == 'vf_loss':
            inset_ax.set_xlim(115, 122)
            inset_ax.set_ylim(0, 0.02)
        elif y_label == 'policy_entropy':
            inset_ax.set_xlim(0, 19)
            inset_ax.set_ylim(-1, 5)
        inset_ax.plot(stats_dict[algo]["x"], stats_dict[algo]["y"], label=algo, color=colors[i])
    axes[0].legend(loc=0)
    axes[1].legend(loc=2)
    axes[2].legend(loc=4)
    axes[0].grid(axis='x')
    axes[0].grid(axis='y')
    axes[0].tick_params(axis='x', labelrotation=90, labelsize=8, labeltop=False, labelbottom=True)
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labeltop=False, labelbottom=False)
    axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labeltop=False, labelbottom=False)

    mark_inset(axes[0], inset_ax, loc1=3, loc2=4)

    plt.show()
    # wandb.init(reinit=True, project="tfm-cm3", name="algo_comparison")
    # wandb.log({y_label: fig})

def log_algo_comparison_metrics(algos):
    episode_reward_mean = {}
    vf_loss = {}
    policy_entropy = {}
    for algo in algos:
        episode_reward_mean[algo] = {"x": [], "y": []}
        vf_loss[algo] = {"x": [], "y": []}
        policy_entropy[algo] = {"x": [], "y": []}
        with open(str(ROOT_DIR) + "/src/stats/comparison/" + algo + "_stats.txt", "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    stats = line.split(',')
                    episode_reward_mean[algo]["x"].append(stats[0])
                    episode_reward_mean[algo]["y"].append(round(float(stats[1]), 2) if stats[1] != 'nan' else 0)
                    if not algo.startswith("a3c"):
                        policy_entropy[algo]["x"].append(stats[0])
                        policy_entropy[algo]["y"].append(round(float(stats[5]), 4) if stats[5] != '' else 0)
                        vf_loss[algo]["x"].append(stats[0])
                        vf_loss[algo]["y"].append(round(float(stats[6]), 4) if stats[6] != '' else 0)

    print("All algorithms' stats collated successfully")
    plot_graph(episode_reward_mean, "episode_reward_mean", algos)
    plot_graph(vf_loss, "vf_loss", algos[:-1])
    plot_graph(policy_entropy, "policy_entropy", algos[:-1])

log_algo_comparison_metrics(["a2c_comparison_486850", "ppo_comparison_262240", "a3c_comparison_936594"])
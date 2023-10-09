from definitions import ROOT_DIR, ITERATION_SEPARATOR
from matplotlib import pyplot as plt
import wandb

def log_and_return_wcs(runNames: []):
    consolidated_wcs = {}
    for runName in runNames:
        blue_wc = []
        purple_wc = []
        itermap_blue = {}
        itermap_purple = {}
        #wandb.init(reinit=True, project="tfm-cm3-winrate", name=runName)
        with open(str(ROOT_DIR) + "/src/wincounts/" + runName + "_win_counts_history.txt", "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                elif line.startswith(ITERATION_SEPARATOR):
                    blue_wins = 0
                    purple_wins = 0
                    for key in itermap_blue.keys():
                        blue_wins += itermap_blue[key]
                    for key in itermap_purple.keys():
                        purple_wins += itermap_purple[key]
                    #wandb.log({"blue_wins_per_iteration": blue_wins, "purple_wins_per_iteration": purple_wins})
                    blue_wc.append(blue_wins)
                    purple_wc.append(purple_wins)
                else:
                    line = line.strip().split(',')
                    worker_keys.add(line[0])
                    itermap_blue[line[0]] = int(line[1])
                    itermap_purple[line[0]] = int(line[2])
        consolidated_wcs[runName] = [blue_wc, purple_wc]
    return consolidated_wcs

runNames = ['qmix_459745', 'mappo_baseline_104890','ppo_stage1_393098','ppo_stage2_882584','ppo_stage3_114265']
worker_keys = set() # just for debugging - should be equal to workers spawned in that run
consolidated_wcs = log_and_return_wcs(runNames)
consolidated_wrs = {}
fig, ax = plt.subplots()
ax.set_ylabel("blue_win_rate")
ax.set_xlabel("iteration")
for runName in runNames:
    consolidated_wrs[runName] = [(n[0] * 100)/(n[0] + n[1]) if n[1] != 0 else 0 for n in zip(*[consolidated_wcs[runName][0], consolidated_wcs[runName][1]])]
final_wrs = {}
final_wrs["qmix"] = consolidated_wrs["qmix_459745"]
final_wrs["mappo_baseline"] = consolidated_wrs["mappo_baseline_104890"]
final_wrs["cm3_2_stage"] = consolidated_wrs["ppo_stage1_393098"][:] + consolidated_wrs["ppo_stage2_882584"][:]
final_wrs["cm3_3_stage"] = consolidated_wrs["ppo_stage1_393098"][:] + consolidated_wrs["ppo_stage2_882584"][0:1000] + consolidated_wrs["ppo_stage3_114265"][:]
plt.plot(final_wrs["qmix"], label="qmix_baseline", color="#A0C75C")
plt.plot(final_wrs["mappo_baseline"], label="mappo_baseline", color="#DA4C4C")
plt.plot(final_wrs["cm3_2_stage"], label="cm3_2_stage", color="#3282F6")
plt.plot(final_wrs["cm3_3_stage"], label="cm3_3_stage", color="#FFD43F", linestyle="dashdot")
plt.legend(loc="upper right")
plt.grid(which="major")
plt.grid(which="minor", linewidth=0.2)
plt.minorticks_on()
plt.show()
wandb.init(reinit=True, project="tfm-cm3-winrate", name="win_rates")
wandb.log({"blue_win_rates": fig})
print("Logged all counts in wandb")
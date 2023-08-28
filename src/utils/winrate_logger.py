from definitions import ROOT_DIR, ITERATION_SEPARATOR
import sys
import wandb

def log_and_return_wcs(runNames: []):
    consolidated_wcs = {}
    for runName in runNames:
        blue_wc = []
        purple_wc = [] #don't really need this, but just to double-check the logic
        itermap_blue = {}
        itermap_purple = {}
        wandb.init(reinit=True, project="ufcm3_winrate", name=runName)
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
                    wandb.log({"blue_wins_per_iteration": blue_wins, "purple_wins_per_iteration": purple_wins})
                    blue_wc.append(blue_wins)
                    purple_wc.append(purple_wins)
                else:
                    line = line.strip().split(',')
                    worker_keys.add(line[0])
                    itermap_blue[line[0]] = int(line[1])
                    itermap_purple[line[0]] = int(line[2])
        consolidated_wcs[runName] = [blue_wc, purple_wc]
    return consolidated_wcs

runNames = sys.argv[1].split(',')
worker_keys = set() #just for debugging - should be equal to workers spawned in that run
consolidated_wcs = log_and_return_wcs(runNames)
print('Logged all counts in wandb')
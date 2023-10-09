from definitions import ROOT_DIR
import wandb


def log_metrics_for_2stage_cm3(stage1, stage2):
    wandb.init(reinit=True, project="tfm-cm3", name="cm3-2stage")
    with open(str(ROOT_DIR) + "/src/stats/" + stage1 + "_stats.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                stats = line.split(',')
                wandb.log({"episode_reward_mean": float(stats[0]),
                           "episode_reward_max": float(stats[1]),
                           "episode_reward_min": float(stats[2]),
                           "policy_loss_blue": float(stats[3]),
                           "policy_entropy_blue": float(stats[4]),
                           "vf_loss_blue": float(stats[5]),
                           "policy_reward_min_blue": None if stats[6] == "" else float(stats[6]),
                           "policy_reward_max_blue": None if stats[7] == "" else float(stats[7]),
                           "policy_reward_mean_blue": None if stats[8] == "" or stats[8] == "\n" else float(stats[8])
                           })
    with open(str(ROOT_DIR) + "/src/stats/" + stage2 + "_stats.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                stats = line.split(',')
                wandb.log({"episode_reward_mean": float(stats[0]),
                           "episode_reward_max": float(stats[1]),
                           "episode_reward_min": float(stats[2]),
                           "policy_loss_blue": float(stats[3]),
                           "policy_entropy_blue": float(stats[4]),
                           "vf_loss_blue": float(stats[5]),
                           "policy_reward_min_blue": None if stats[6] == "" else float(stats[6]),
                           "policy_reward_max_blue": None if stats[7] == "" else float(stats[7]),
                           "policy_reward_mean_blue": None if stats[8] == "" or stats[8] =="\n" else float(stats[8])
                           })


def log_metrics_for_3stage_cm3(stage1, stage2, stage3):
    wandb.init(reinit=True, project="tfm-cm3", name="cm3-3stage")
    with open(str(ROOT_DIR) + "/src/stats/" + stage1 + "_stats.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                stats = line.split(',')
                wandb.log({"episode_reward_mean": float(stats[0]),
                           "episode_reward_max": float(stats[1]),
                           "episode_reward_min": float(stats[2]),
                           "policy_loss_blue": float(stats[3]),
                           "policy_entropy_blue": float(stats[4]),
                           "vf_loss_blue": float(stats[5]),
                           "policy_reward_min_blue": None if stats[6] == "" else float(stats[6]),
                           "policy_reward_max_blue": None if stats[7] == "" else float(stats[7]),
                           "policy_reward_mean_blue": None if stats[8] == "" or stats[8] == "\n" else float(stats[8])
                           })
    with open(str(ROOT_DIR) + "/src/stats/" + stage2 + "_stats.txt", "r") as f:
        iter_count = 0
        while iter_count < 1000:
            line = f.readline()
            if not line:
                break
            else:
                stats = line.split(',')
                wandb.log({"episode_reward_mean": float(stats[0]),
                           "episode_reward_max": float(stats[1]),
                           "episode_reward_min": float(stats[2]),
                           "policy_loss_blue": float(stats[3]),
                           "policy_entropy_blue": float(stats[4]),
                           "vf_loss_blue": float(stats[5]),
                           "policy_reward_min_blue": None if stats[6] == "" else float(stats[6]),
                           "policy_reward_max_blue": None if stats[7] == "" else float(stats[7]),
                           "policy_reward_mean_blue": None if stats[8] == "" or stats[8] =="\n" else float(stats[8])
                           })
            iter_count += 1
    with open(str(ROOT_DIR) + "/src/stats/" + stage3 + "_stats.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                stats = line.split(',')
                wandb.log({"episode_reward_mean": float(stats[0]),
                           "episode_reward_max": float(stats[1]),
                           "episode_reward_min": float(stats[2]),
                           "policy_loss_blue": float(stats[3]),
                           "policy_entropy_blue": float(stats[4]),
                           "vf_loss_blue": float(stats[5]),
                           "policy_reward_min_blue": None if stats[6] == "" else float(stats[6]),
                           "policy_reward_max_blue": None if stats[7] == "" else float(stats[7]),
                           "policy_reward_mean_blue": None if stats[8] == "" or stats[8] =="\n" else float(stats[8])
                           })


log_metrics_for_2stage_cm3("ppo_stage1_393098", "ppo_stage2_882584")
log_metrics_for_3stage_cm3("ppo_stage1_393098", "ppo_stage2_882584", "ppo_stage2_2_548873")
import json, random, ray, time, wandb, sys, logging
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray import tune
from datetime import datetime
from src.utils.unity3d_env_wrapper import Unity3DEnv
from src.utils.custom_side_channel import CustomSideChannel
from definitions import ROOT_DIR
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

algo_config = sys.argv[1]
asset_name = sys.argv[2]

logging.basicConfig(level=logging.DEBUG, filename=datetime.now().strftime(__file__ + "_comparison_%d_%m_%Y_%H_%M_%S.log"), filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

asset_file_name = str(ROOT_DIR) + "/src/assets/" + asset_name + "/UnityEnvironment.exe"
NO_GRAPHICS_MODE = True

logging.info("Building assets from: " + asset_name)
wandb.init(project="tfm-cm3", name="ppo_comparison")

random_filename_appender = random.randint(100000, 999999)
stats_filename = "ppo_comparison_" + str(random_filename_appender)
logging.info("Tracking wincounts in " + stats_filename + "_win_counts_history.txt")
customChannel = CustomSideChannel(stats_filename)

config_file = open(str(ROOT_DIR) + "/src/configs/" + algo_config + ".json")
run_config = json.load(config_file)
config_file.close()

env_name = "SoccerTwosRR"
logging.info("Extracting policies for env_name: " + env_name)
policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(env_name)

tune.register_env(
        env_name,
        lambda c: Unity3DEnv(
            file_name=asset_file_name,
            episode_horizon=3000,
            no_graphics=NO_GRAPHICS_MODE,
            side_channels=[customChannel]
),)

config = {
    "env": env_name,
    "env_config": {
                "file_name": asset_file_name,
                "episode_horizon": 3000,
    },
    "disable_env_checking": True,
    "framework": "tf",
    "model": {
        "fcnet_hiddens": run_config['fcnet_hiddens']
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn
    },
    "lambda": float(run_config['lambda']),
    "gamma": float(run_config['gamma']),
    "lr": float(run_config['lr']),
    "sgd_minibatch_size": int(run_config['sgd_minibatch_size']),
    "train_batch_size": int(run_config['train_batch_size']),
    "clip_param": float(run_config['clip_param']),
    "num_rollout_workers": int(run_config['num_rollout_workers']),
    "num_cpus_per_worker": int(run_config['num_cpus_per_worker'])
}

# below only for running in GCP instances
# ray.init(num_cpus=1)
trainer = PPOTrainer(config=config)

iter = 0
env_steps = 0

while env_steps < int(run_config['env_steps']):
    iter += 1
    logging.info("Training iteration " + str(iter))
    start = time.time()
    result = trainer.train()
    end = time.time()
    customChannel.checkpoint_win_counts(stats_filename)
    env_steps = result['num_env_steps_trained']
    logging.info("Iteration " + str(iter) + " finished in " + str(end-start) + "s, # of env_steps trained " + str(env_steps))
    # Save stats in a local file only as wandb doesn't allow plotting (x, y)
    # Use this file afterwards to plot a matplotlib graph > export to wandb
    with open(str(ROOT_DIR) + "/src/stats/" + stats_filename + "_stats.txt", "a+") as stats_file:
        stats_file.write(str(env_steps) + ","
                         + str(result['episode_reward_mean']) + ","
                         + str(result['episode_reward_max']) + ","
                         + str(result['episode_reward_min']) + ","
                         + str(result['info']['learner']['BluePlayer']['learner_stats']['policy_loss']) + ","
                         + str(result['info']['learner']['BluePlayer']['learner_stats']['entropy']) + ","
                         + str(result['info']['learner']['BluePlayer']['learner_stats']['vf_loss']) + ","
                         + (str(result['policy_reward_min']['BluePlayer']) if result['policy_reward_min'] else "") + ","
                         + (str(result['policy_reward_max']['BluePlayer']) if result['policy_reward_min'] else "") + ","
                         + (str(result['policy_reward_mean']['BluePlayer']) if result['policy_reward_min'] else "")
                         + "\n")
        stats_file.close()
    # below just to track the run remotely
    wandb.log({"steps_trained": env_steps})

logging.info("Shutting down ray")
ray.shutdown()
import json, random, ray, time, wandb, sys, logging, tensorflow as tf
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.policy.policy import Policy
from ray import tune
from datetime import datetime
from src.utils.unity3d_env_wrapper import Unity3DEnv
from src.utils.custom_side_channel import CustomSideChannel
from definitions import ROOT_DIR
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
NO_GRAPHICS_MODE = False
logging.basicConfig(level=logging.DEBUG, filename=datetime.now().strftime(__file__ + "_play_%d_%m_%Y_%H_%M_%S.log"), filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

algo_config = sys.argv[1]
asset_name = sys.argv[2]
blue_checkpoint_path = sys.argv[3]
purple_checkpoint_path = sys.argv[4]

asset_file_name = str(ROOT_DIR) + "/src/assets/" + asset_name + "/UnityEnvironment.exe"
logging.info("Building assets from: " + asset_name)

random_filename_appender = random.randint(100000, 999999)
stats_filename = "ppo_play_" + str(random_filename_appender)
logging.info("Tracking wincounts in " + stats_filename + "_win_counts_history.txt")
customChannel = CustomSideChannel(stats_filename)

config_file = open(str(ROOT_DIR) + "/src/configs/" + algo_config + ".json")
run_config = json.load(config_file)
config_file.close()

env_name = "SoccerTwosRR-Play" # gets both the policies
logging.info("Extracting policies for env_name: " + env_name)
policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(env_name)

tune.register_env(
        env_name,
        lambda c: Unity3DEnv(
            file_name=asset_file_name,
            episode_horizon=3000,
            no_graphics=True,
            side_channels=[customChannel]
),)
env = Unity3DEnv(
            file_name=asset_file_name,
            episode_horizon=3000,
            no_graphics=NO_GRAPHICS_MODE,
            side_channels=[customChannel]
)

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

trainer = PPOTrainer(config=config)

blue_checkpoint_path = blue_checkpoint_path + "/policies/BluePlayer"
purple_checkpoint_path = purple_checkpoint_path + "/policies/BluePlayer"
logging.info("Updating BluePlayer policy weights from path: " + blue_checkpoint_path + ", and PurplePlayer policy weights from: " + purple_checkpoint_path)
blue_pre_trained_policy = Policy.from_checkpoint(blue_checkpoint_path)
purple_pre_trained_policy = Policy.from_checkpoint(purple_checkpoint_path)
trainer.set_weights({"BluePlayer": blue_pre_trained_policy.get_weights(), "PurplePlayer": purple_pre_trained_policy.get_weights()})
local_weights = trainer.workers.local_worker().get_weights()
trainer.workers.foreach_worker(lambda worker: worker.set_weights(local_weights))
logging.info("Weights updated for all workers")

episode = 0

while episode < int(run_config['play_episode_count']):
    episode += 1
    logging.info("Play episode " + str(episode))
    obs = env.reset()
    blue_ep_reward = 0
    purple_ep_reward = 0
    start = time.time()
    terminated = False
    while not terminated:
        pa1 = trainer.compute_single_action(obs["SoccerTwos?team=1_0"], policy_id="PurplePlayer")
        pa2 = trainer.compute_single_action(obs["SoccerTwos?team=1_2"], policy_id="PurplePlayer")
        ba1 = trainer.compute_single_action(obs["SoccerTwos?team=0_1"], policy_id="BluePlayer")
        ba2 = trainer.compute_single_action(obs["SoccerTwos?team=0_3"], policy_id="BluePlayer")
        obs, reward, done, info = env.step({"SoccerTwos?team=1_0": pa1, "SoccerTwos?team=1_2": pa2, "SoccerTwos?team=0_1": ba1, "SoccerTwos?team=0_3": ba2})
        terminated = done["__all__"]
        blue_ep_reward += (reward["SoccerTwos?team=0_1"] + reward["SoccerTwos?team=0_3"])
        purple_ep_reward += (reward["SoccerTwos?team=1_0"] + reward["SoccerTwos?team=1_2"])

    end = time.time()
    customChannel.checkpoint_win_counts(stats_filename)
    logging.info("Episode " + str(episode) + " finished in " + str(end-start) + "s")

    with open(str(ROOT_DIR) + "/src/stats/" + stats_filename + "_stats.txt", "a+") as stats_file:
        stats_file.write(str(blue_ep_reward) + "," + str(purple_ep_reward) + "\n")
        stats_file.close()

logging.info("Shutting down ray")
ray.shutdown()
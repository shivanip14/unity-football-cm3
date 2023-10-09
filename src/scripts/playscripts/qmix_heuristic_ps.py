import json, random, ray, time, sys, logging

import numpy as np
from ray import tune
from datetime import datetime
from gym.spaces import Tuple, Discrete, Box
from src.utils.unity3d_env_wrapper_qmix import Unity3DEnvQMix
from ray.rllib.algorithms.qmix import QMixConfig, QMix as QmixTrainer
from src.utils.custom_side_channel import CustomSideChannel
from definitions import ROOT_DIR
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
NO_GRAPHICS_MODE = True
logging.basicConfig(level=logging.DEBUG, filename=datetime.now().strftime(__file__ + "_play_%d_%m_%Y_%H_%M_%S.log"), filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

algo_config = sys.argv[1]
asset_name = sys.argv[2]
blue_checkpoint_path = sys.argv[3]

asset_file_name = str(ROOT_DIR) + "/src/assets/" + asset_name + "/UnityEnvironment.exe"
logging.info("Building assets from: " + asset_name)

random_filename_appender = random.randint(100000, 999999)
stats_filename = "qmix_heuristic_play_" + str(random_filename_appender)
logging.info("Tracking wincounts in " + stats_filename + "_win_counts_history.txt")
customChannel = CustomSideChannel(stats_filename)

config_file = open(str(ROOT_DIR) + "/src/configs/" + algo_config + ".json")
run_config = json.load(config_file)
config_file.close()

def groupedenv(args):
    o = Tuple([Box(-1.0, 1.0, (264,)), Box(-1.0, 1.0, (72,)),])
    a = Discrete(27)
    obs_space = Tuple([o, o])
    act_space = Tuple([a, a])
    grouping = {'team_0': ['SoccerTwos?team=0?agent_id=1', 'SoccerTwos?team=0?agent_id=3'],
                'team_1': ['SoccerTwos?team=1?agent_id=0', 'SoccerTwos?team=1?agent_id=2'], }
    sa_env = Unity3DEnvQMix(file_name=asset_file_name,
                            episode_horizon=3000,
                            no_graphics=NO_GRAPHICS_MODE,
                            side_channels=[customChannel])
    grouped_env = sa_env.with_agent_groups(groups=grouping, obs_space=obs_space, act_space=act_space)
    return grouped_env

tune.register_env("grouped_SoccerTwos", groupedenv)

config = (QMixConfig()
          .environment(disable_env_checking=True,
                       env="grouped_SoccerTwos")
          .framework("torch")
          .training(mixer=run_config['mixer'],
                    double_q=(run_config['double_q'] == "True"))
          .training(model={"fcnet_hiddens": run_config['fcnet_hiddens'],
                           "max_seq_len": int(run_config['max_seq_len'])},
                    gamma=float(run_config['gamma']),
                    lr=float(run_config['lr']),
                    train_batch_size=int(run_config['train_batch_size'])))
config._lambda = float(run_config['lambda'])
config.simple_optimizer = (run_config['simple_optimizer'] == "True")
config.horizon = int(run_config['horizon'])
config.sgd_minibatch_size = int(run_config['sgd_minibatch_size'])

# below only for running in local for debugging
ray.init(local_mode=True)
trainer = QmixTrainer(config=config)

env = groupedenv(1)

logging.info("Updating policy weights from path: " + blue_checkpoint_path)
trainer.restore(blue_checkpoint_path)
local_weights = trainer.workers.local_worker().get_weights()
trainer.workers.foreach_worker(lambda worker: worker.set_weights(local_weights))
logging.info("Weights updated for all workers")

episode = 0

while episode < int(run_config['play_episode_count']):
    episode += 1
    logging.info("Play episode " + str(episode))
    obs = env.reset()
    state = trainer.get_state()
    blue_ep_reward = 0
    purple_ep_reward = 0
    start = time.time()
    terminated = False
    while not terminated:
        flattened_obs = np.concatenate(np.concatenate(obs["team_0"]).ravel()).ravel()
        ba, rnn_states, extra = trainer.get_policy().compute_single_action(obs=flattened_obs, state=trainer.get_policy().get_initial_state())
        obs, reward, done, info = env.step({"SoccerTwos?team=0?agent_id=1": ba[0], "SoccerTwos?team=0?agent_id=3": ba[1]})
        terminated = done["__all__"]

    end = time.time()
    customChannel.checkpoint_win_counts(stats_filename)
    logging.info("Episode " + str(episode) + " finished in " + str(end-start) + "s")

logging.info("Shutting down ray")
ray.shutdown()
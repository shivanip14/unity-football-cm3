import json, random, ray, time, wandb, sys, logging, tensorflow as tf
from ray.rllib.algorithms.qmix import QMixConfig, QMix as QmixTrainer
from ray import tune
from datetime import datetime
from gym.spaces import Tuple, Discrete, Box
from src.utils.unity3d_env_wrapper_qmix import Unity3DEnvQMix
from src.utils.custom_side_channel import CustomSideChannel
from definitions import ROOT_DIR

logging.basicConfig(level=logging.DEBUG, filename=datetime.now().strftime(__file__ + "_%d_%m_%Y_%H_%M_%S.log"),
                    filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

curr_os = sys.argv[1]
algo_config = sys.argv[2]
asset_name = sys.argv[3]

asset_file_name = str(ROOT_DIR) + "/src/assets/" + asset_name + "/UnityEnvironment.exe"
NO_GRAPHICS_MODE = False
if curr_os == 'linux':
    asset_file_name = str(ROOT_DIR) + "/src/assets/" + asset_name + "-linux/UnityEnvironment.x86_64"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    NO_GRAPHICS_MODE = True

logging.info("Building assets from: " + asset_name)
# wandb.init(project="ufcm3")

random_filename_appender = random.randint(100000, 999999)
stats_filename = "qmix_" + str(random_filename_appender)
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
# ray.init(local_mode=True)
trainer = QmixTrainer(config=config)

iter = 0

while iter < int(run_config['run_iter_count']):
    iter += 1
    logging.info("Training iteration " + str(iter))
    start = time.time()
    result = trainer.train()
    end = time.time()
    customChannel.checkpoint_win_counts(stats_filename)
    logging.info("Iteration " + str(iter) + " finished in " + str(end - start) + "s")
    with open(str(ROOT_DIR) + "/src/stats/" + stats_filename + "_stats.txt", "a+") as stats_file:
        stats_file.write(str(result['episode_reward_mean']) + "\n")
        stats_file.close()
    # wandb.log({"episode_reward_mean": result['episode_reward_mean']})
    if iter % 10 == 0:
        checkpoint_dir = trainer.save()
        logging.info('Checkpoint after iteration ' + str(iter) + ' saved in directory ' + str(checkpoint_dir))

logging.info("Shutting down ray")
ray.shutdown()

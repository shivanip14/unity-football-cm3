import json, random, ray, time, wandb, sys, logging, tensorflow as tf
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.policy.policy import Policy
from ray import tune
from datetime import datetime
from src.utils.unity3d_env_wrapper import Unity3DEnv
from src.utils.custom_side_channel import CustomSideChannel
from definitions import ROOT_DIR

logging.basicConfig(level=logging.DEBUG, filename=datetime.now().strftime(__file__ + "_%d_%m_%Y_%H_%M_%S.log"), filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

curr_os = sys.argv[1]
algo_config = sys.argv[2]
asset_name = sys.argv[3]
# stage 0 = non-CL run, default
stage_of_run = int(sys.argv[4]) if len(sys.argv) > 4 else 0
checkpoint_path = sys.argv[5] if stage_of_run > 1 else ""

asset_file_name = str(ROOT_DIR) + "/src/assets/" + asset_name + "/UnityEnvironment.exe"
NO_GRAPHICS_MODE = False
if curr_os == 'linux':
    asset_file_name = str(ROOT_DIR) + "/src/assets/" + asset_name + "-linux/UnityEnvironment.x86_64"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    NO_GRAPHICS_MODE = True

logging.info("Building assets from: " + asset_name)
#wandb.init(project="ufcm3")

random_filename_appender = random.randint(100000, 999999)
stats_filename = "ppo_stage" + str(stage_of_run) + "_" + str(random_filename_appender)
logging.info("Tracking wincounts in " + stats_filename + "_win_counts_history.txt")
customChannel = CustomSideChannel(stats_filename)

config_file = open(str(ROOT_DIR) + "/src/configs/" + algo_config + ".json")
run_config = json.load(config_file)
config_file.close()

# og 2v2 version without random roles, i.e. without goals as an input to policy network
env_name = "SoccerTwos"
if stage_of_run == 1:
    # RR = with random roles added as goals to policy network, SA = single agent version
    env_name = "SoccerTwosRR-SA"
elif stage_of_run > 1:
    # RR = with random roles added as goals to policy network, 2v2 multi-agent version
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
    "framework": "tf2",
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

if stage_of_run > 1:
    checkpoint_path = checkpoint_path + "/policies/BluePlayer"
    logging.info("Updating BluePlayer policy weights from path: " + checkpoint_path)
    pre_trained_policy = Policy.from_checkpoint(checkpoint_path)
    trainer.set_weights({"BluePlayer": pre_trained_policy.get_weights()})

iter = 0

while iter < int(run_config['run_iter_count']):
    iter += 1
    logging.info("Training iteration " + str(iter))
    start = time.time()
    result = trainer.train()
    end = time.time()
    customChannel.checkpoint_win_counts(stats_filename)
    logging.info("Iteration " + str(iter) + " finished in " + str(end-start) + "s")
    with open(str(ROOT_DIR) + "/src/stats/" + stats_filename + "_stats.txt", "a+") as stats_file:
        stats_file.write(str(result['episode_reward_mean']) + ","
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
    # wandb.log({"episode_reward_mean": result['episode_reward_mean'],
    #            "episode_reward_max": result['episode_reward_max'],
    #            "episode_reward_min": result['episode_reward_min'],
    #            "policy_loss_blue": result['info']['learner']['BluePlayer']['learner_stats']['policy_loss'],
    #            "policy_entropy_blue": result['info']['learner']['BluePlayer']['learner_stats']['entropy'],
    #            "vf_loss_blue": result['info']['learner']['BluePlayer']['learner_stats']['vf_loss'],
    #            "policy_reward_min_blue": result['policy_reward_min']['BluePlayer'] if result['policy_reward_min'] else None,
    #            "policy_reward_max_blue": result['policy_reward_max']['BluePlayer'] if result['policy_reward_min'] else None,
    #            "policy_reward_mean_blue": result['policy_reward_mean']['BluePlayer'] if result['policy_reward_min'] else None
    #            })
    if iter % 10 == 0:
        checkpoint_dir = trainer.save()
        logging.info('Checkpoint after iteration ' + str(iter) + ' saved in directory ' + str(checkpoint_dir))

logging.info("Shutting down ray")
ray.shutdown()
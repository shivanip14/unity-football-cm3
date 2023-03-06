import ray
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray import tune
import wandb

wandb.init(
  # set the wandb project where this run will be logged
  project="just-tinkering",
  name="a2c-unity-football-sa-8",
  # track hyperparameters and run metadata
  config={
    "environment": "Unity-SoccerTwos",
    "epochs": 100,
    "config_lambda": 0.95,
    "config_gamma": 0.99,
    "config_lr": 0.0001,
    "config_entropy_coeff": 0.001,
    "config_train_batch_size": 5000,
    "training_model": "fcnets_hidden_512_512"
  }
)

policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game("SoccerPlayer")

tune.register_env(
        "SoccerTwos",
        lambda c: Unity3DEnv(
            file_name="/src/assets/single-agent/UnityEnvironment.exe",
            episode_horizon=3000,
            no_graphics=True
),)

config = (
    A2CConfig()
    .environment("SoccerTwos",
            env_config={
                "file_name": "D:/MAI/TFM/codebases/unity-football-cm3/src/assets/single-agent/UnityEnvironment.exe",
                "episode_horizon": 3000,
            },
            disable_env_checking=True)
    .framework("tf2")
    .training(model={"fcnet_hiddens": [512, 512]})
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
)
config.lambda_ = 0.95
config.gamma = 0.99
config.lr = 0.0001
config.entropy_coeff = 0.001
config.train_batch_size = 5000

print("Building")
algo = config.build()
print("Build complete")

for i in range(100):
    print("Train loop " + str(i))
    result = algo.train()
    print(result)
    wandb.log({"episode_reward_mean": result['episode_reward_mean'],
               "episode_reward_max": result['episode_reward_max'],
               "episode_reward_min": result['episode_reward_min'],
               "policy_loss": result['info']['learner']['SoccerPlayer']['learner_stats']['policy_loss'],
               "policy_entropy": result['info']['learner']['SoccerPlayer']['learner_stats']['policy_entropy'],
               "vf_loss": result['info']['learner']['SoccerPlayer']['learner_stats']['vf_loss'],
               "policy_reward_min": result['policy_reward_min']['SoccerPlayer'] if result['policy_reward_min'] else None,
               "policy_reward_max": result['policy_reward_max']['SoccerPlayer'] if result['policy_reward_min'] else None,
               "policy_reward_mean": result['policy_reward_mean']['SoccerPlayer'] if result['policy_reward_min'] else None
               })
    if i % 10 == 0:
        checkpoint_dir = algo.save()
        print('Checkpoint after episode ' + str(i) + ' saved in directory ' + str(checkpoint_dir))

# print("Evaluating")
# algo.evaluate()
print("Shutting down ray")
ray.shutdown()
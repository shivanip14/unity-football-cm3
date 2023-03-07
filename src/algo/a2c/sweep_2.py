import ray
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray import tune
import wandb

def train_sweep(config=None):
    with wandb.init(name="a2c-unity-football-sweep-2", config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game("SoccerPlayer")

        tune.register_env(
            "SoccerTwos",
            lambda c: Unity3DEnv(
                file_name="D:/MAI/TFM/codebases/unity-football-cm3/src/assets/single-agent/UnityEnvironment.exe",
                episode_horizon=3000,
                no_graphics=True
            ), )

        algo_config = (
            A2CConfig()
            .environment("SoccerTwos",
                         env_config={
                             "file_name": "D:/MAI/TFM/codebases/unity-football-cm3/src/assets/single-agent/UnityEnvironment.exe",
                             "episode_horizon": 3000,
                         },
                         disable_env_checking=True)
            .framework("tf2")
            .training(model={
                "fcnet_hiddens": [config.fcnet_hidden_1, config.fcnet_hidden_2],
                "fcnet_activation": config.fcnet_activation,
                "lstm_use_prev_action": config.lstm_use_prev_action,
                "lstm_use_prev_reward": config.lstm_use_prev_reward,
                "max_seq_len": config.max_seq_len,
                "use_lstm": config.use_lstm,
                "vf_share_layers": config.vf_share_layers
            })
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        )
        algo_config.lambda_ = config.lambda_
        algo_config.gamma = config.gamma
        algo_config.lr = config.lr
        algo_config.entropy_coeff = config.entropy_coeff
        algo_config.vf_loss_coeff = config.vf_loss_coeff
        algo_config.train_batch_size = config.train_batch_size
        algo_config.use_critic = config.use_critic
        algo_config.use_gae = config.use_gae
        algo_config.rollout_fragment_length = config.rollout_fragment_length

        print("Building")
        algo = algo_config.build()
        print("Build complete")

        for i in range(1):
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

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'policy_entropy',
        'goal': 'minimize'
    },
    'parameters': {
        'fcnet_activation': {
            'values': ['relu', 'tanh']
        },
        'fcnet_hidden_1': {
            'values': [128, 256, 512, 1024]
        },
        'fcnet_hidden_2': {
            'values': [128, 256, 512, 1024]
        },
        'max_seq_len': {
            'distribution': 'uniform',
            'min': 10,
            'max': 30
        },
        'lr': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'lambda_': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1.0
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1.0
        },
        'entropy_coeff': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
        'train_batch_size': {
            'min': 4000,
            'max': 6000
        },
        'vf_loss_coeff': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.8
        },
        'rollout_fragment_length': {
            'values': [10, 50, 100]
        },
        'use_critic': {
            'values': [True, False]
        },
        'use_gae': {
            'values': [True, False]
        },
        'use_lstm': {
            'values': [True, False]
        },
        'vf_share_layers': {
            'values': [True, False]
        },
        'lstm_use_prev_action': {
            'values': [True, False]
        },
        'lstm_use_prev_reward': {
            'values': [True, False]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="just-tinkering-spam")
print('Starting sweep ' + str(sweep_id))
wandb.agent(sweep_id=sweep_id, function=train_sweep, count=20)
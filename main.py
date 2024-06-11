from stable_baselines3.common.env_checker import check_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env import DynamicCrimeEnv
from agents import FixedUniformAgent, SamplingUniformAgent, StaticMLEAgent, DynamicMLEAgent, FairAgent

import logging

# TODO: remove
# # Disable pymc logging
# LOG_PYMC = False
# if LOG_PYMC == False:
#     logger = logging.getLogger("pymc.sampling")
#     logger.propagate = False

MAX_STEPS = 50
SEED = 42

ENV_CONFIG = {
    "initial_lambdas": [11.35, 27.44, 20.37, 7.36, 22.67, 10.47, 17.26, 19.83, 30.97, 28.69, 43.5, 17.36, 17.41, 25.88, 33.43, 30.45, 38.47, 35.54, 20.55, 30.92, 23.24],
    "dynamic_factor": 0.0,
    "resources": 500,
    "max_crimes": 80,
    "max_t": MAX_STEPS
}

AGENT_CONFIG_BASIC = {
    "resources": ENV_CONFIG["resources"],
    "areas": len(ENV_CONFIG["initial_lambdas"]),
}

AGENT_CONFIG_SOPH = {
    "resources": ENV_CONFIG["resources"],
    "areas": len(ENV_CONFIG["initial_lambdas"]), 
    "burnin_steps": 10,
    "max_crimes": ENV_CONFIG["max_crimes"],
    "mle_interval": 2,
}


env = DynamicCrimeEnv(**ENV_CONFIG)
# agent = DynamicMLEAgent(**AGENT_CONFIG)
agents = [
    FixedUniformAgent(**AGENT_CONFIG_BASIC), 
    FixedUniformAgent(**AGENT_CONFIG_BASIC), # To double check seed, TODO: remove
    SamplingUniformAgent(**AGENT_CONFIG_BASIC), 
    SamplingUniformAgent(**AGENT_CONFIG_BASIC),
    StaticMLEAgent(**AGENT_CONFIG_SOPH),
    StaticMLEAgent(**AGENT_CONFIG_SOPH),
    # DynamicMLEAgent(**AGENT_CONFIG_SOPH, dynamic_factor=ENV_CONFIG["dynamic_factor"]),
    FairAgent(**AGENT_CONFIG_SOPH, alpha=0.05),
    FairAgent(**AGENT_CONFIG_SOPH, alpha=0.05)
]

agents_names = [
    'fixed',
    'fixed_copy',
    'sampling_uniform',
    'sampling_uniform_copy',
    'static_mle',
    'static_mle_copy',
    # 'dynamic_mle',
    'fair',
    'fair_copy'
]

dataframes = []

for agent, agent_name in zip(agents, agents_names):
    obs, info = env.reset(seed=SEED)
    agent.seed(SEED)
    terminated = False

    dataframe_dict = {
        "prevented_crimes": [],
        "committed_crimes": [],
        "accuracy": [],
        "equality_discovery_prob": [],
        "equality_wellness": [],
        # "true_lambdas": [],
    }

    for t in range(MAX_STEPS):
        print(f"Agent {agent_name}, timestep {t}")
        action, info_agent = agent.act(obs, t)

        obs, reward, terminated, truncated, info_env = env.step(action)

        for key in dataframe_dict.keys():
            dataframe_dict[key].append(info_env[key])

        if terminated:
            print(f"Episode terminated for agent {agent_name}")

    dataframe = pd.DataFrame.from_dict(dataframe_dict)
    dataframe.to_pickle(f"df_{agent_name}")

    dataframes.append(dataframe)


fig = plt.figure()

for df, label in zip(dataframes, agents_names):
    plt.plot(np.arange(MAX_STEPS), df['equality_discovery_prob'], label=label)

plt.title("Equality of discovery probability")
plt.legend(loc="upper right")
plt.show()

for df, label in zip(dataframes, agents_names):
    plt.plot(np.arange(MAX_STEPS), df['accuracy'], label=label)

plt.title("Accuracy")
plt.legend(loc="upper right")
plt.show()

# TODO: plot means of each field in a barplot







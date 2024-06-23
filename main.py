import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from env import DynamicCrimeEnv
from agents import SamplingUniformAgent, StaticMLEAgent, FairAgent, SteeringAgent

# Command line arguments
if len(sys.argv) < 2:
    print("""
          Usage: python main.py <city_name>
            city_name: the name of the city to simulate (philadelphia, los_angeles)
          """)
    sys.exit(1)

CITY_NAME = sys.argv[1]
if CITY_NAME not in ["philadelphia", "los_angeles"]:
    print("Invalid city name. Choose one of the following: philadelphia or los_angeles")
    sys.exit(1)


# the philadelphia lambdas come from the paper "Fair Algorithms for Learning in Allocation Problems"
# the los_angeles lambdas are computed using the notebook ./data/lambdas_exctractor.ipynb
LAMBDAS_DB = {
    "philadelphia": [11.35, 27.44, 20.37, 7.36, 22.67, 10.46, 17.25, 19.82, 30.96, 28.68, 43.49, 17.35, 17.40, 25.87, 33.42, 30.44, 38.46, 35.53, 20.54, 30.91, 23.23],
    "los_angeles": [28.06, 39.67, 33.22, 25.05, 30.72, 29.58, 28.88, 23.48, 27.28, 24.81, 26.71, 29.38, 21.58, 24.19, 24.96, 36.7, 34.2, 29.79, 24.24, 19.54, 24.29],
}

### Configuration ###
# Simulation
MAX_STEPS = 700
BURNIN_STEPS = 30
SEED = 42
DYNAMIC_FACTOR = 0.008

# Data
LAMBDAS = LAMBDAS_DB[CITY_NAME]

# Fair agents
ALPHA = 0.05
STEERING_FACTOR = 5
EXPLOITING_RANGE = 0.15
#
###


### Static simulation ###
ENV_CONFIG = {
    "initial_lambdas": LAMBDAS,
    "dynamic_factor": 0.0,
    "resources": int(round(sum(LAMBDAS))),
    "max_crimes": int(round(max(LAMBDAS) * 2)),
    "max_t": MAX_STEPS
}

print(f"Total resources: {ENV_CONFIG['resources']}")
print(f"Max crimes: {ENV_CONFIG['max_crimes']}")

AGENT_CONFIG_BASIC = {
    "resources": ENV_CONFIG["resources"],
    "areas": len(ENV_CONFIG["initial_lambdas"]),
}

AGENT_CONFIG_SOPH = {
    "resources": ENV_CONFIG["resources"],
    "areas": len(ENV_CONFIG["initial_lambdas"]), 
    "burnin_steps": BURNIN_STEPS,
    "max_crimes": ENV_CONFIG["max_crimes"],
    "mle_interval": 1,
}


env = DynamicCrimeEnv(**ENV_CONFIG)
agents = [
    SamplingUniformAgent(**AGENT_CONFIG_BASIC), 
    StaticMLEAgent(**AGENT_CONFIG_SOPH),
    FairAgent(**AGENT_CONFIG_SOPH, alpha=ALPHA),
]

agents_names = [
    'random_uniform',
    'max_util',
    'fair',
]

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
    }

    print(f"Starting simulation for agent {agent_name} and environment with dynamic factor {ENV_CONFIG['dynamic_factor']}")
    for t in tqdm(range(MAX_STEPS)):
        action, info_agent = agent.act(obs, t)

        obs, reward, terminated, truncated, info_env = env.step(action)

        for key in dataframe_dict.keys():
            dataframe_dict[key].append(info_env[key])

    dataframe = pd.DataFrame.from_dict(dataframe_dict)
    filename = f"./outputs/{CITY_NAME}/df_{agent_name}"
    dataframe.to_pickle(filename)

    print(f"Simulation dataframe saved in {filename}")

###


### Dynamic simulation ###

ENV_CONFIG['dynamic_factor'] = DYNAMIC_FACTOR


env = DynamicCrimeEnv(**ENV_CONFIG)
agents = [
    SamplingUniformAgent(**AGENT_CONFIG_BASIC), 
    StaticMLEAgent(**AGENT_CONFIG_SOPH),
    FairAgent(**AGENT_CONFIG_SOPH, alpha=ALPHA),
    SteeringAgent(**AGENT_CONFIG_SOPH, alpha=ALPHA, steering_factor=STEERING_FACTOR, exploiting_range=EXPLOITING_RANGE),
]

agents_names = [
    'random_uniform',
    'max_util',
    'fair',
    'steering',
]

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
    }

    print(f"Starting simulation for agent {agent_name} and environment with dynamic factor {ENV_CONFIG['dynamic_factor']}")
    for t in tqdm(range(MAX_STEPS)):
        action, info_agent = agent.act(obs, t)

        obs, reward, terminated, truncated, info_env = env.step(action)

        for key in dataframe_dict.keys():
            dataframe_dict[key].append(info_env[key])

    dataframe = pd.DataFrame.from_dict(dataframe_dict)
    filename = f"./outputs/{CITY_NAME}/df_{agent_name}_dynamic"
    dataframe.to_pickle(filename)

    print(f"Simulation dataframe saved in {filename}")





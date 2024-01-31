from typing import Sequence, Tuple

from dm_control.locomotion import soccer as dm_soccer
from dm_control import suite
from dm_control.rl import control
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optax
import reverb
import rlax
import tensorflow as tf

import acme
from acme import specs
from acme import wrappers
from acme.adders import reverb as reverb_adders
from acme.agents.jax import actors
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.d4pg import learning
from acme.datasets import reverb as datasets
from acme.jax import utils, variable_utils
from acme.jax import networks as networks_lib
from acme.jax.experiments.run_experiment import _disable_insert_blocking, _LearningActor
from acme.utils import counting
from acme.utils import loggers

# from IPython.display import HTML
from dm_control import viewer
import pdb

### Configure the D4PG agent

key = jax.random.PRNGKey(123)

batch_size = 256
learning_rate = 1e-4
discount = 0.99
n_step = 5  # The D4PG agent learns from n-step transitions.
exploration_sigma = 0.1
target_update_period = 100

# Controls the relative rate of sampled vs inserted items. In this case, items
# are n-step transitions.
samples_per_insert = 32.0

# Atoms used by the categorical distributional critic.
num_atoms = 51
critic_atoms = jnp.linspace(-150., 150., num_atoms)

### Load the environment
environment: control.Environment = dm_soccer.load(team_size=1,
                                    time_limit=10.0,
                                    random_state=0,
                                    disable_walker_contacts=False,
                                    enable_field_box=True,
                                    terminate_on_goal=True,
                                    walker_type=dm_soccer.WalkerType.NAO)
                                    #  walker_type=dm_soccer.WalkerType.HUMANOID)
# pdb.set_trace()
name_filter = ['joints_pos', 'team_goal_front_left', 'team_goal_back_right', 'ball_ego_position']

# Concatenate the observations (position, velocity, etc).
environment = wrappers.ConcatObservationWrapper(environment, name_filter=name_filter)

# Make the environment expect continuous action spec is [-1, 1].
# Note: this is a no-op on dm_control tasks.
# environment = wrappers.CanonicalSpecWrapper(environment, clip=True)

# # Make the environment output single-precision floats.
# # We use this because most TPUs only work with float32.
environment = wrappers.SinglePrecisionWrapper(environment)

# Environment specs
environment_spec = specs.make_environment_spec(environment)
# pdb.set_trace()

### Create the Haiku networks
# Calculate how big the last layer should be based on total # of actions.
action_spec = environment_spec.actions
action_size = np.prod(action_spec.shape, dtype=int)

# Create the deterministic policy network.
def policy_fn(obs: networks_lib.Observation) -> jnp.ndarray:
  x = obs
  x = networks_lib.LayerNormMLP([256, 256], activate_final=True)(x)
  x = networks_lib.NearZeroInitializedLinear(action_size)(x)
  x = networks_lib.TanhToSpec(action_spec)(x)
  return x

# Create the distributional critic network.
def critic_fn(
    obs: networks_lib.Observation,
    action: networks_lib.Action,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  x = jnp.concatenate([obs, action], axis=-1)
  x = networks_lib.LayerNormMLP(layer_sizes=[256, 256, num_atoms])(x)
  return x, critic_atoms

policy = hk.without_apply_rng(hk.transform(policy_fn))
critic = hk.without_apply_rng(hk.transform(critic_fn))


### Dummy Environment
# Create dummy observations and actions to create network parameters.
dummy_action = utils.zeros_like(environment_spec.actions)
dummy_obs = utils.zeros_like(environment_spec.observations)

# Prebind dummy observations and actions so they are not needed in the learner.
policy_network = networks_lib.FeedForwardNetwork(
    init=lambda rng: policy.init(rng, dummy_obs),
    apply=policy.apply)
critic_network = networks_lib.FeedForwardNetwork(
    init=lambda rng: critic.init(rng, dummy_obs, dummy_action),
    apply=critic.apply)


def exploration_policy(
    params: networks_lib.Params,
    key: networks_lib.PRNGKey,
    observation: networks_lib.Observation,
) -> networks_lib.Action:
  action = policy_network.apply(params, observation)
  if exploration_sigma:
    action = rlax.add_gaussian_noise(key, action, exploration_sigma)
  return action
     
     
### Create a D4PG agent components
# central counter
parent_counter = counting.Counter(time_delta=0.)

# replay tables
# Manages the data flow by limiting the sample and insert calls.
rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
    min_size_to_sample=1000,
    samples_per_insert=samples_per_insert,
    error_buffer=2 * batch_size)

# Create a replay table to store previous experience.
replay_tables = [
    reverb.Table(
        name='priority_table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1_000_000,
        rate_limiter=rate_limiter,
        signature=reverb_adders.NStepTransitionAdder.signature(
            environment_spec))
]


# NOTE: This is the first of three code cells that are specific to
# single-process execution. (This is done for you when you use an agent
# `Builder` and `run_experiment`.) Everything else is logic shared between the
# two.
replay_tables, rate_limiters_max_diff = _disable_insert_blocking(replay_tables)



replay_server = reverb.Server(replay_tables, port=None)
replay_client = reverb.Client(f'localhost:{replay_server.port}')



# Pull data from the Reverb server into a TF dataset the agent can consume.
dataset = datasets.make_reverb_dataset(
    table='priority_table',
    server_address=replay_client.server_address,
    batch_size=batch_size,
)

# We use multi_device_put here in case this colab is run on a machine with
# multiple accelerator devices, but this works fine with single-device learners
# as long as their step functions are pmapped.
dataset = utils.multi_device_put(dataset.as_numpy_iterator(), jax.devices())

# NOTE: This is the second of three code cells that are specific to
# single-process execution. (This is done for you when you use an agent
# `Builder` and `run_experiment`.) Everything else is logic shared between the
# two.
dataset = utils.prefetch(dataset, buffer_size=1)


### Create the learner
key, learner_key = jax.random.split(key)

# The learner updates the parameters (and initializes them).
learner = learning.D4PGLearner(
    policy_network=policy_network,
    critic_network=critic_network,
    random_key=learner_key,
    policy_optimizer=optax.adam(learning_rate),
    critic_optimizer=optax.adam(learning_rate),
    discount=discount,
    target_update_period=target_update_period,
    iterator=dataset,
    # A simple counter object that can periodically sync with a parent counter.
    counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.),
)


### Create the adder
# Handles preprocessing of data and insertion into replay tables.
adder = reverb_adders.NStepTransitionAdder(
    priority_fns={'priority_table': None},
    client=replay_client,
    n_step=n_step,
    discount=discount)

### Create the actor
key, actor_key = jax.random.split(key)

# A convenience adaptor from FeedForwardPolicy to ActorCore.
actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
    exploration_policy)

# A variable client for updating variables from a remote source.
variable_client = variable_utils.VariableClient(learner, 'policy', device='cpu')
actor = actors.GenericActor(
    actor=actor_core,
    random_key=actor_key,
    variable_client=variable_client,
    adder=adder,
    backend='cpu')

# NOTE: This is the third of three code cells that are specific to
# single-process execution. (This is done for you when you use an agent
# `Builder` and `run_experiment`.) Everything else is logic shared between the
# two.
actor = _LearningActor(actor, learner, dataset, replay_tables,
                       rate_limiters_max_diff, checkpointer=None)

env_loop_logger = loggers.InMemoryLogger()

# Create the environment loop used for training.
env_loop = acme.EnvironmentLoop(
    environment,
    actor,
    counter=counting.Counter(parent_counter, prefix='train', time_delta=0.),
    logger=env_loop_logger)


### Run a training loop
env_loop.run(num_episodes=50)
df = pd.DataFrame(env_loop_logger.data)
plt.figure(figsize=(10, 4))
plt.title('Training episodes returns')
plt.xlabel('Training episodes')
plt.ylabel('Episode return')
plt.plot(df['episode_return'])
plt.savefig("training.png")

### Run and visualize the agent in the environment
# Make the environment render frames and produce videos of episodes.
eval_environment = wrappers.MujocoVideoWrapper(environment, camera_id=1, height=480, width=640, record_every=1, path="/home/leyong/projects/humanoid_robotic_systems/hrs_ws/src/dm_control_scripts", filename="training")

timestep = eval_environment.reset()

while not timestep.last():
  action = actor.select_action(timestep.observation)
  print("action:", action)
  timestep = eval_environment.step(action)

eval_environment.close()

# Embed the HTML video.
# HTML(eval_environment.make_html_animation())
     
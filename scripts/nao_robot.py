import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from acme import wrappers
import pdb

env = dm_soccer.load(team_size=1,
                     time_limit=10.0,
                     random_state=0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=True,
                     walker_type=dm_soccer.WalkerType.NAO)
                    #  walker_type=dm_soccer.WalkerType.HUMANOID)


# viewer.launch(env)
# pdb.set_trace()
action_specs = env.action_spec()
print(action_specs)


### Run and visualize the agent in the environment
# Make the environment render frames and produce videos of episodes.
eval_environment = wrappers.MujocoVideoWrapper(env, camera_id=1, height=480, width=640, record_every=1, path="/home/leyong/projects/humanoid_robotic_systems/hrs_ws/src/dm_control_scripts", filename="robot")

timestep = eval_environment.reset()

# viewer.launch(eval_environment)
# pdb.set_trace()
i = 0
while not timestep.last():
  # actions = np.random.uniform(action_specs.minimum, action_specs.maximum, size=action_specs.shape)
  action = np.array([0, 	-1.53, 0])
  print("actions:", action)
  print("joints_pos: ", timestep.observation['joints_pos'])
  timestep = eval_environment.step(action)
  i += 1
  print("i:", i)
  print("time step:", i * 0.025)
  

viewer.launch(env)
eval_environment.close()
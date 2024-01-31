from dm_control.locomotion import soccer as dm_soccer
from dm_control import mjcf
from dm_control import viewer
import pdb

env = dm_soccer.load(team_size=1,
                     time_limit=10.0,
                     random_state=0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=True,
                     walker_type=dm_soccer.WalkerType.NAO)

timestep = env.reset()
# viewer.launch(env)
pdb.set_trace()
# mjcf.export_with_assets(env.task.players[0].walker._mjcf_root, "./nao_balancing")
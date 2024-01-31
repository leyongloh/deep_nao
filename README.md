# deep_nao
Reinforcement learning for NAO robot using DeepMind dm_control and acme packages. Please check the commit history of both cloned repo for a throughly understanding of the changes.

# Installation 
Please refers to dm_control and acme for complete installation guide. The requirement.txt is there to ensure compability. Tested on python3==3.9.7, if you are using 3.8 or 3.10 you might run into some errors.
As the installation is quite tedious because of a lots of compatability issues, please email me if you need help.

# Scripts
this contains the python scripts for training (aceme_pipeline.py), conversion of mjcf class to xml (convert_mjcf_to_xml.py) and visualizing the environment (nao_robot.py).

# hrs_ws
this folder is the hrs workspace and contain the classical Nao robot control, since the devcontainer uses python2, I would recommend to run the dm_control and acme in a seperate python environment with python 3.9.7 to avoid conflicts.


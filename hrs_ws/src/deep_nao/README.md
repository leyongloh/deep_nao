# How to run
To run exercise 1:
```python
roslaunch nao_control_tutorial_2 nao.launch Ex:="Ex1"
```

To run exercise 2:
```python
roslaunch nao_control_tutorial_2 nao.launch or roslaunch nao_control_tutorial_2 nao.launch Ex:="Ex2"
```

To run exercise 2.2, print the manually computed homogeneous matrix (CameraOpticalFrame -> CameraBottom):
```python
roslaunch nao_control_tutorial_2 nao.launch Ex:="Ex2.2"
```

# Useful commands
source /opt/ros/kinetic/setup.bash
source devel/setup.bash
roslaunch nao_bringup nao_full_py.launch
rosrun nao_control_tutorial_2 move_service.py 10.152.246.248 9559
rosrun nao_control_tutorial_2 move_client.py Ex1
rosrun nao_control_tutorial_2 move_client.py Ex2
rosrun nao_control_tutorial_2 move_client.py Ex2.2
rosservice call /body_stiffness/disable "{}"
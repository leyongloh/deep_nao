# How to run DeepNao: Classic Robot Control

source /opt/ros/kinetic/setup.bash
source devel/setup.bash
roslaunch nao_bringup nao_full_py.launch
rosrun deep_nao move_service.py 10.152.246.123 9559
roslaunch nao_apps tactile.launch
python src/deep_nao/script/perception_ball.py
python src/deep_nao/script/ball_subscriber.py
python src/deep_nao/script/aruco_marker_subscriber.py 
python src/deep_nao/script/move2ball.py

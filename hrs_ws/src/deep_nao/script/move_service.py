#!/usr/bin/env python
import rospy
import time
import motion
import almath
import sys
from naoqi import ALProxy
from deep_nao.srv import MoveJoints, MoveJointsResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import copy
import numpy as np
import tf
import pdb
motionProxy = 0
postureProxy = 0

def callback(req):
    end_effector = req.end_effector
    frame = req.frames
    pose = req.pose
    fraction_max_velocity = req.fraction_max_velocity
    execution_time = req.execution_time
    position_only = req.position_only


    # Get the current position of the robot
    useSensorValues = True
    cartesian_position = motionProxy.getPosition(end_effector[0], frame, useSensorValues)
    print("MoveService: Original Position of", end_effector, "in Torso Frame is:", cartesian_position)


    # # check position limits
    if pose.position.x > 0.15:
        pose.position.x = 0.15
    if pose.position.y > 0.15:
        pose.position.y = 0.15
    if pose.position.z > 0.15:
        pose.position.z = 0.15

    # Extract the destinated positions out of the message
    destination = [pose.position.x, pose.position.y,pose.position.z,
                   pose.orientation.x,pose.orientation.y,pose.orientation.z]
    
    # 
    if position_only:
        axis_mask = 7
    else:
        axis_mask = 63

    # 
    if execution_time ==  ():
        # Set the position 
        print("MoveService: setPositions")
        motionProxy.setPositions(end_effector, frame, destination, fraction_max_velocity, axis_mask)
    else:
        print("MoveService: positionInterpolations")
        if len(end_effector) >= 2:
            # if we need to move more than 1 joint then execute this lines
            destination = [destination, destination]
            axis_mask = [axis_mask, axis_mask]
            execution_time = [execution_time, execution_time]
        motionProxy.positionInterpolations(end_effector, frame, destination, axis_mask, execution_time)

    time.sleep(0.5)

    cartesian_position = motionProxy.getPosition(end_effector[0], frame, useSensorValues)
    print("MoveService: Goal Position of", end_effector, "in Torso Frame is:", cartesian_position)
    return MoveJointsResponse(cartesian_position)

def move_service():
    rospy.init_node("move_joints_serverNode", anonymous=True)
    s = rospy.Service("move_serviceServer", MoveJoints, callback)
    while not rospy.is_shutdown():
        # Initialize the robot to standing position
        postureProxy.goToPosture("Stand", 1.0)
        rospy.spin()

    if rospy.is_shutdown():
        # set robot to rest positon and stiffness to 0.0 
        motionProxy.rest()
        motionProxy.setStiffnesses("Head", 0.0)
        motionProxy.setStiffnesses("LArm", 0.0)
        motionProxy.setStiffnesses("RArm", 0.0)
        motionProxy.setStiffnesses("LHand", 0.0)
        motionProxy.setStiffnesses("RHand", 0.0)

if __name__ == '__main__':
    robotIP=str(sys.argv[1])
    PORT=int(sys.argv[2])
    motionProxy = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    move_service()
			


		

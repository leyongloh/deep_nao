#!/usr/bin/env python
import motion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
import numpy as np
import tf
import rospy
import time
import almath
import sys
from naoqi import ALProxy
from deep_nao.srv import MoveJoints
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from std_msgs.msg import String
import pdb
import math
from visualization_msgs.msg import Marker
from naoqi_bridge_msgs.msg import Bumper
from naoqi_bridge_msgs.msg import WordRecognized

postureProxy = 0
motionProxy = 0

class Move2Ball:
    def __init__(self):
        rospy.init_node('ball_perception_listener', anonymous=True)
        
        # Ball Pose Subscriber
        self.ball_sub = rospy.Subscriber("/ball_pose", Pose, self.callback_ball, queue_size=1)
        # Aruco Marker Subscriber
        self.aruco_sub = rospy.Subscriber("/aruco_marker_position", Marker, self.callback_aruco, queue_size=1)
        # Bumper Subscriber
        self.bumper_sub = rospy.Subscriber("/bumper", Bumper, self.bumper_callback)
        
        #self.sub_speech = rospy.Subscriber("/word_speech", String, self.speech_callback)
        
        self.ball_pose_x = None
        self.ball_pose_y = None
        self.ball_pose_z = None
        
        self.left_bumper_active = None
        
        self.aruco_2_pose_y = 0
        self.aruco_1_pose_y = 0
        
        self.speech_word = 'start'
        
    # Bumper callback
    def bumper_callback(self, data):
        if data.bumper == Bumper.left:
            # Reset the robot to StandInit Position
            self.left_bumper_active = data.state == 1
            print("Left bumper active")
            postureProxy.goToPosture("StandInit", 0.9)
    
    # def speech_callback(self, data):
    #     self.speech_word = data
    #     print("Received Word:", self.speech_word)
        
    # Ball position callback
    def callback_ball(self, msg):
        # Read Pose message
        self.ball_pose_x = msg.position.x
        self.ball_pose_y = msg.position.y
        self.ball_pose_z = msg.position.z
        print("call back ball_pose_x:", self.ball_pose_x)
        
    # Aruco marker callback
    def callback_aruco(self, msg):
        # Read Marker Id and position 
        self.aruco_marker_id = msg.id
        print("aruco_marker_id", self.aruco_marker_id)
        
        if self.aruco_marker_id == 1:            
            self.aruco_1_pose_x = msg.pose.position.x
            self.aruco_1_pose_y = msg.pose.position.y
            self.aruco_1_pose_z = msg.pose.position.z
            
        elif self.aruco_marker_id == 2:            
            self.aruco_2_pose_x = msg.pose.position.x
            self.aruco_2_pose_y = msg.pose.position.y
            self.aruco_2_pose_z = msg.pose.position.z
            
        distance_y_aruco_1_2 = math.sqrt((self.aruco_2_pose_y-self.aruco_1_pose_y)**2)
        print("distance_y_aruco_1_2", distance_y_aruco_1_2)
        
        # self.move_robot_to_ball()
        # Check if goal is in front of Nao
        print("ball pose received:", self.ball_pose_x)
        if distance_y_aruco_1_2 < 1.0:
            # Execute move2ball action
            self.move_robot_to_ball()
        #else:
            #self.reset_robot()


    # Function to move the robot by sending the messages to move_service server
    def move_robot_to_ball(self):

        if self.ball_pose_x and self.ball_pose_y:
            print("self.ball_pose_x", self.ball_pose_x )
            print("self.ball_pose_y", self.ball_pose_y )
            #print("self.aruc_pose_x", self.aruc_pose_x )
        
            # Move the left hip roll joint on the same height as the ball
        
            a = self.ball_pose_y  / self.ball_pose_x
            lhip_roll_angle = math.atan(a)
            print("lhip_roll_angle:", lhip_roll_angle)
            #motionProxy.setAngles("LHipRoll",lhip_roll_angle, 0.1)
            
            # Speech recognition
            #pdb.set_trace()
            # if self.speech_word == "shoot":
            #     pass
            # else:
            #     self.reset_robot()
            
            rospy.sleep(5)
    
            ## Kicking movement
        
            # Detect if the ball is in front of the foot and if the aruco marker is in range
            if self.ball_pose_y < 0.05 and self.ball_pose_y > 0.01 and self.ball_pose_x < 0.25:
                # if self.ball_pose_x < 0.25:
                print("Perception listener: Execute Kick")
                motionProxy.setAngles("LHipRoll",lhip_roll_angle, 0.1)
                rospy.sleep(5)
                
                # Freeze every joint
                motionProxy.setStiffnesses("Body", 1.0)
                motionProxy.setStiffnesses("LLeg", 1.0)
                
                # Move left hip pitch joint - Kicking Motion
                motionProxy.setAngles("LHipPitch", -0.8, 1.0)
                
                rospy.sleep(5)
                # self.reset_robot()
                print("End kick")
        
    def reset_robot(self):
        # reset robot to original standing configuration
        print("ball_perception_listener: RESET")
        postureProxy.goToPosture("StandInit", 0.9)
        
    def start(self):   
        rospy.spin()
    
def init():
    # Initialize the robot
    postureProxy.goToPosture("StandInit", 1.0)
    
    rospy.sleep(2)
    
    motionProxy.moveInit()
    motionProxy.wbEnable(True)
    motionProxy.wbGoToBalance("RLeg", 3.0)
    motionProxy.setStiffnesses("LLeg", 0.0)
    
    print(motionProxy.getAngles("RHipPitch", True))
    print(motionProxy.getAngles("RKneePitch", True))
    print(motionProxy.getAngles("RAnklePitch", True))
    print(motionProxy.getAngles("RHipRoll", True))
    
    motionProxy.setAngles("RHipRoll", -0.05, 1.0)
    
    rospy.sleep(4)
    
    hiproll= motionProxy.getAngles("RHipRoll", True)
    print(motionProxy.getAngles("RHipRoll", True))
    
    rospy.sleep(4)
    
    motionProxy.setAngles("HeadPitch", 0.5, 0.1)
    
    rospy.sleep(4)

if __name__ == '__main__':

    robotIP = "10.152.246.123"
    PORT = 9559
    
    # posture and motion proxy
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    motionProxy = ALProxy("ALMotion", robotIP, PORT)
    
    init()

    move2ball = Move2Ball()
    move2ball.start()
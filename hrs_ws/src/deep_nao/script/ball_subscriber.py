#!/usr/bin/env python3
import rospy
import os
import numpy as np 
import pdb
import math
import tf
from naoqi import ALProxy
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import Header


class Ball_Pose_Calculator:
    def __init__(self):
        # Publisher
        self.pub_ball_pose = rospy.Publisher('/ball_pose', Pose, queue_size=10)
        
        # Sub ball pose from ball perception
        self.sub_ball_pose = rospy.Subscriber('/ball_pose_raw',Pose, self.callback, queue_size=5)

        # Subscribe Torso and Bottom camera pose
        self.torso_pose_sub = rospy.Subscriber('/torso_pose', PoseStamped, self.callback, queue_size=5)
        self.cam_bottom_pose_sub = rospy.Subscriber("/CameraBottom_frame", PoseStamped, self.callback, queue_size=5)
        
        self.ball_center_image_x = None
        self.y_ball_camera = None
        self.z_foot_camera = None
    
    def callback(self, msg):
        listener = tf.TransformListener()
        
        # Center of  image frame
        output_frame_x = 320/2
        output_frame_y = 240/2
        
        if msg.position.x != 0:
            # Calucalte ball pose from image data
            ball_center_x = int(msg.position.x)
            print("ball_center_x:", ball_center_x)
            ball_center_y = int(msg.position.y)
            print("ball_center_y:", ball_center_y)

            # Offset of the ball position in perspective to image center
            self.ball_center_image_x = -(ball_center_x - output_frame_x)*0.001
            print("ball_center_image_x", self.ball_center_image_x)
            ball_center_image_y = (ball_center_y - output_frame_y)*0.001 
            print("ball_center_image_y", ball_center_image_y)

            # Angle of camera to torso
            listener.waitForTransform('/CameraBottom_frame', 'torso', rospy.Time(), rospy.Duration(1.0)) 
            cam_bot_listener = self.torso_pose_sub
            cam_bot_listener = PoseStamped()
            cam_bot_listener.header.frame_id = "/CameraBottom_frame"

            cam_torso_trans = listener.transformPose("/torso", cam_bot_listener)
            pitch_cam_torso = cam_torso_trans.pose.orientation.y

            print("pitch_cam_torso", pitch_cam_torso)


            # Distance from bottom Camera to ground
            listener.waitForTransform('/l_sole', 'CameraBottom_frame', rospy.Time(), rospy.Duration(1.0))
            self.cam_bottom_pose_sub = PoseStamped()
            self.cam_bottom_pose_sub.header.frame_id = "/CameraBottom_frame"

            cam_torso_trans = listener.transformPose("/l_sole", self.cam_bottom_pose_sub)
            z_cam_bottom = cam_torso_trans.pose.position.z

            print("z_cam_bottom", z_cam_bottom)


            # Foot position 
            listener.waitForTransform('/torso', '/l_sole', rospy.Time(), rospy.Duration(1.0))
            self.torso_pose_sub = PoseStamped()
            self.torso_pose_sub.header.frame_id = "/l_sole"

            l_sole_pose = listener.transformPose("/torso", self.torso_pose_sub)
            self.z_foot_camera = l_sole_pose.pose.position.z

            print("z_foot_camera", self.z_foot_camera)


            # Distance from ball to Bottom Camera
            self.y_ball_camera = - ball_center_image_y + z_cam_bottom*math.atan(0.5)
            

            # Publish ball pose
            tf_transformBroadcaster = tf.TransformBroadcaster()
            tf_transformBroadcaster.sendTransform((self.y_ball_camera, self.ball_center_image_x, self.z_foot_camera), 
                                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                                rospy.Time.now(), "ball_pose", "torso")
          
            ball_pose_msg = Pose()
            ball_pose_msg.position.x = self.y_ball_camera 
            ball_pose_msg.position.y = self.ball_center_image_x
            ball_pose_msg.position.z = self.z_foot_camera
            ball_pose_msg.orientation.x = 0
            ball_pose_msg.orientation.y = 0
            ball_pose_msg.orientation.z = 0
            ball_pose_msg.orientation.w = 1
            self.pub_ball_pose.publish(ball_pose_msg)

            print("-----------------------------------------")
            print("Rviz position of the ball")
            print("Ball x offset:", self.y_ball_camera)
            print("ball y offset:", self.ball_center_image_x)
            print("Ball z offset:", self.z_foot_camera)
            print("-----------------------------------------")
        
    def start(self):   
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.init_node("ball_pose", anonymous=True)
        ball_pose_calculator = Ball_Pose_Calculator()
        ball_pose_calculator.start()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python
import rospy
import time
import almath
import sys
from naoqi import ALProxy
from nao_control_tutorial_2.srv import MoveJoints
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from std_msgs.msg import String
import pdb

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import copy
import numpy as np
import math

import motion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose2D
import tf
motionProxy =0
postureProxy =0


class Nodo(object):
    def __init__(self):
        # Params
        rospy.init_node("imagetimer111", anonymous=True)
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(200)

        # Subscribers
        self.image_sub= rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.callback) 

        # Parameters for calculating joint angles
        self.image_center_x = None
        self.image_center_y = None

        # distance parameters
        self.distance_x = None
        self.distance_y = None
        self.distance_z = None

        # Homogeneous transformation matrices
        self.H_camerabottomoptical_camerabottom = None
        self.H_camera_robot_correction = None
        # self.H_test = NonepostureProxy
        self.motionProxy = ALProxy("ALMotion", robotIP, PORT)

        self.pub = rospy.Publisher('aruco_marker_position',Point, latch=True)

    def callback(self, msg):
        rospy.loginfo('Image received...')

        self.image = self.br.imgmsg_to_cv2(msg, desired_encoding ="bgr8")
        aruco_image = copy.deepcopy(self.image)

        # Aruco marker detection
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners_aruco, ids_aruco, rejected_aruco) = cv2.aruco.detectMarkers(aruco_image, arucoDict, parameters=arucoParams)
        # ids_aruco: (left, 1), (right, 2)

        # print("ids_aruco[0]: ", ids_aruco[0])
        # print("ids_aruco[1]: ", ids_aruco[1])
        # center of the image
        height, weight, _ = self.image.shape
        self.image_center_x = weight / 2
        self.image_center_y = height / 2

        # aruco marker size for determining depth
        markerSizeInCM = 0.1

        # camera calibration parameters
        # mtx = np.array([[ 551.543059,    0., 327.382898],
        #                 [    0., 553.736023, 225.026380],
        #                 [    0.,    0.,           1.]])
        
        # dist = np.array([[-0.066494], [0.095481], [-0.000279], [0.002292], [0.000000]])
        # TODO: HERE WE USE CAMERABOTTOM parameters but the image is from CAMERATOP
        mtx = np.array([[ 278.236008818534,    0., 156.194471689706],
                        [    0., 279.380102992049, 126.007123836447],
                        [    0.,    0.,           1.]])
        
        dist = np.array([[-0.0481869853715082], [0.0201858398559121], [0.0030362056699177], [-0.00172241952442813], [0.000000]])

        if (corners_aruco != []):
            for k, (corners, ids) in enumerate(zip(corners_aruco, ids_aruco)):
                cv2.aruco.drawDetectedMarkers(aruco_image, corners_aruco, ids_aruco)
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
                average_corners_aruco = corners.mean(axis=1)
                x,y = average_corners_aruco[0]
                cv2.circle(aruco_image, (x,y), 2, (60,255,255), 3) # plot aruco marker center
                cv2.circle(aruco_image, (self.image_center_x, self.image_center_y), 2, (60,255,255), 3) # plot image center

                # extract x,y,z position of aurco marker
                self.distance_x = tvec[0][0][0]
                self.distance_y = tvec[0][0][1]
                self.distance_z = tvec[0][0][2]

                self.publish_aruco() # publish the aruco wrt. CameraOpticalFrame
                self.CameraTop_optical_to_CameraTop() # get Homogeneoue Transformation CameraOpticalFrame to CameraBottom
                self.aruco_marker_wrt_torsoframe(ids) # get aruco position wrt to torso

        else:
            # if aruco marker is not detected send zero positions
            if (self.distance_x != 0) and (self.distance_y !=0) and (self.distance_z !=0):
                self.distance_x = 0
                self.distance_y = 0
                self.distance_z = 1
                self.p_aruco[0] = 0
                self.p_aruco[1] = 0
                self.p_aruco[2] = 0

        # Publish the aruco marker position
        self.pub.publish(self.p_aruco[0], self.p_aruco[1], self.p_aruco[2])

    # Function to broadcast the aruco marker position wrt. CameraOpticalFrame
    def publish_aruco(self):
        tf_transformBroadcaster = tf.TransformBroadcaster()
        tf_transformBroadcaster.sendTransform((self.distance_x, self.distance_y, self.distance_z), 
                                              tf.transformations.quaternion_from_euler  (0, 0, 0),
                                              rospy.Time.now(), "aruco_marker", "CameraBottom_optical_frame")


    # Ex2.2 Compute Manually the static homogeneous transformation of
    # CameraBottom_optical frame to CameraBottom frame
    def CameraTop_optical_to_CameraTop(self):
        Rz = np.array([[np.cos(1.5708), -np.sin(1.5708), 0],
                       [np.sin(1.5708), np.cos(1.5708), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(0), 0, np.sin(0)],
                       [0, 1, 0],
                       [-np.sin(0), 0, np.cos(0)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(1.5708), -np.sin(1.5708)],
                       [0, np.sin(1.5708), np.cos(1.5708)]])
        R = np.dot(np.dot(Rx, Ry), Rz)
        self.H_camerabottomoptical_camerabottom = np.array([[R[0,0], R[0,1], R[0,2], 0],[R[1,0],R[1,1], R[1,2], 0], [R[2,0], R[2,1], R[2,2], 0], [0,0,0,1]])
        # print("Rotation matrix R:", R)
        # print("Homogeneous transformation of CameraBottom_optical to CameraBottom: ", self.H_camerabottomoptical_camerabottom)

    # Transform the aruco marker from CameraOpticalFrame to Torso
    def aruco_marker_wrt_torsoframe(self, ids):
        useSensorValues = False
        H_result = self.motionProxy.getTransform("CameraTop", motion.FRAME_TORSO, useSensorValues)

        # Homogeneous Transformation of camerabottom wrt. torso
        H_camerabottom_torso = np.array([[H_result[0], H_result[1], H_result[2], H_result[3]],
                                         [H_result[4], H_result[5], H_result[6], H_result[7]],
                                         [H_result[8], H_result[9], H_result[10], H_result[11]],
                                         [H_result[12], H_result[13], H_result[14], H_result[15]]])
        
        # aurco marker initial position
        p_aruco = np.array([self.distance_x, self.distance_y, self.distance_z, 1])
        p_aruco = np.dot(p_aruco, self.H_camerabottomoptical_camerabottom) # aruco marker wrt. camerabottom
        p_aruco = np.dot(p_aruco, H_camerabottom_torso) # aruco marker wrt. torso
        self.p_aruco = p_aruco
        # print("p_aruco:", self.p_aruco)

        # aruco marker ids
        tf_frame_name = "aruco_marker_wrt_torso_" + str(ids)
        z_offset = -0.625
        x_offset = 0.3
        # if ids[0] == 1: 
        #     z_offset = 1.35
        # elif ids[0] == 2:   
        #     z_offset = 1.15

        # Broadcast the aruco marker wrt. torso to /tf topic
        tf_transformBroadcaster_torso = tf.TransformBroadcaster()
        tf_transformBroadcaster_torso.sendTransform((self.p_aruco[0] + x_offset, self.p_aruco[1], self.p_aruco[2] + z_offset), 
                                              tf.transformations.quaternion_from_euler(0, 0, 0),
                                              rospy.Time.now(), tf_frame_name, "torso")

    def start(self):
        rospy.loginfo("Timing images")
        while not rospy.is_shutdown():
            self.loop_rate.sleep()

def aruco_ex2():
    # Initializes aruco marker detection Node
    aruco_marker_node = Nodo()
    aruco_marker_node.start()

def stand_rest():
    motionProxy.rest()
    rospy.sleep(5)
    postureProxy.goToPosture("Stand", 1.0)
    rospy.sleep(5)
    motionProxy.rest()
    rospy.sleep(5)
    postureProxy.goToPosture("Stand", 1.0)
    rospy.sleep(5)
    motionProxy.rest()
    rospy.sleep(5)
    postureProxy.goToPosture("Stand", 1.0)

def walking():
    rospy.init_node('nao_walk_control')
    walk_pub = rospy.Publisher('/cmd_pose', Pose2D, latch=True)
    postureProxy.goToPosture("Stand", 1.0)
    rospy.sleep(4)
    goal_position = Pose2D(x=1.5, y=0.0,theta=0.0)
    walk_pub.publish(goal_position)
    rospy.sleep(10)

if __name__ == '__main__':
    robotIP = "10.152.246.123"
    PORT = 9559
    motionProxy = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    
    # postureProxy.goToPosture("Stand", 1.0)
    # # walking()
    # # stand_rest()
    motionProxy.rest()
    rospy.sleep(5)
    
    #while not rospy.is_shutdown():

    postureProxy.goToPosture("StandInit", 1.0)
    # motionProxy.setStiffnesses("RLeg", 0.0)
    # postureProxy.goToPosture("SitRelax", 1.0)
    # postureProxy.goToPosture("LyingBack", 1.0)
    # # postureProxy.goToPosture("LyingBelly", 1.0)
    rospy.sleep(5)
    # motionProxy.angleInterpolation("RHipPitch", -0.0, 2.0, True)
    # motionProxy.angleInterpolation("RHipPitch", -0.0, 2.0, True)
    motionProxy.setStiffnesses("Body", 1.0)
    motionProxy.moveInit()
    motionProxy.wbEnable(True)
    motionProxy.wbGoToBalance("LLeg", 10.0)
    motionProxy.setStiffnesses("RLeg", 0.0)
    # motionProxy.setStiffnesses("RLeg", 0.0)
    rospy.sleep(7)
    # motionProxy.setStiffnesses("LLeg", 0.0)
    # rospy.sleep(4)
    # motionProxy.setStiffnesses("LLeg", 1.0)
    # cartesian_position = motionProxy.getPosition(end_effector[0], frame, True)
    # print("Position of", end_effector, "in Torso Frame is:", cartesian_position)
    # motionProxy.setStiffnesses("Head", 1.0)
    # rospy.sleep(5)
    # postureProxy.goToPosture("LyingBelly", 1.0)
    # postureProxy.goToPosture("LyingBack", 1.0)
    # motionProxy.setStiffnesses("Head", 1.0)
    # # motionProxy.angleInterpolation("HeadPitch", 0.0, 2.0, True)
    # motionProxy.angleInterpolation("HeadPitch", 0.5, 2.0, True)
    # motionProxy.setStiffnesses("Head", 0.0)
    end_effector = ["LArm"]
    frame = motion.FRAME_TORSO
    cartesian_position = motionProxy.getPosition(end_effector[0], frame, True)
    print("MoveService: Original Position of", end_effector, "in Torso Frame is:", cartesian_position)
    # aruco_ex2()
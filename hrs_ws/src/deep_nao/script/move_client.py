#!/usr/bin/env python
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

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import copy
import numpy as np
import math

import motion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

import tf


class Nodo(object):
    def __init__(self):
        # Params
        rospy.init_node("imagetimer111", anonymous=True)
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(200)

        # Subscribers
        self.image_sub= rospy.Subscriber("/nao_robot/camera/bottom/camera/image_raw",Image,self.callback) 

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
        self.H_test = None
        self.p_aruco = np.zeros(4)

        # Motion Proxy
        robotIP = "10.152.246.248"
        PORT = 9559
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

        # center of the image
        height, weight, _ = self.image.shape
        self.image_center_x = weight / 2
        self.image_center_y = height / 2

        # aruco marker size for determining depth
        markerSizeInCM = 0.088

        # camera calibration parameters
        mtx = np.array([[ 278.236008818534,    0., 156.194471689706],
                        [    0., 279.380102992049, 126.007123836447],
                        [    0.,    0.,           1.]])
        
        dist = np.array([[-0.0481869853715082], [0.0201858398559121], [0.0030362056699177], [-0.00172241952442813], [0.000000]])

        if (corners_aruco != []):
            cv2.aruco.drawDetectedMarkers(aruco_image, corners_aruco, ids_aruco)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners_aruco, markerSizeInCM, mtx, dist)
            average_corners_aruco = corners_aruco[0].mean(axis=1)
            x,y = average_corners_aruco[0]
            cv2.circle(aruco_image, (x,y), 2, (60,255,255), 3) # plot aruco marker center
            cv2.circle(aruco_image, (self.image_center_x, self.image_center_y), 2, (60,255,255), 3) # plot image center

            # extract x,y,z position of aurco marker
            self.distance_x = tvec[0][0][0]
            self.distance_y = tvec[0][0][1]
            self.distance_z = tvec[0][0][2]

            self.publish_aruco() # publish the aruco wrt. CameraOpticalFrame
            self.CameraBottom_optical_to_CameraBottom() # get Homogeneoue Transformation CameraOpticalFrame to CameraBottom
            self.aruco_marker_wrt_torsoframe() # get aruco position wrt to torso

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
                                              tf.transformations.quaternion_from_euler(0, 0, 0),
                                              rospy.Time.now(), "aruco_marker", "CameraBottom_optical_frame")


    # Ex2.2 Compute Manually the static homogeneous transformation of
    # CameraBottom_optical frame to CameraBottom frame
    def CameraBottom_optical_to_CameraBottom(self):
        R_z_reflection = np.array([[1,0,0],
                                   [0,1,0],
                                   [0,0,-1]])
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
    def aruco_marker_wrt_torsoframe(self):
        useSensorValues = False
        H_result = self.motionProxy.getTransform("CameraBottom", motion.FRAME_TORSO, useSensorValues)

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

        # Broadcast the aruco marker wrt. torso to /tf topic
        tf_transformBroadcaster_torso = tf.TransformBroadcaster()
        tf_transformBroadcaster_torso.sendTransform((self.p_aruco[0], self.p_aruco[1], self.p_aruco[2]), 
                                              tf.transformations.quaternion_from_euler(0, 0, 0),
                                              rospy.Time.now(), "aruco_marker_wrt_torso", "torso")

    def start(self):
        rospy.loginfo("Timing images")
        while not rospy.is_shutdown():
            self.loop_rate.sleep()


def move_ex1():
    end_effector = ["LArm"]
    frame = motion.FRAME_TORSO

    # Position and Orientations
    pose = Pose()
    pose.position.x = 0.12465044111013412
    pose.position.y = 0.2412523776292801
    pose.position.z = 0.17257067561149597
    pose.orientation.x = 3.1392879486083984
    pose.orientation.y = -0.6281214356422424
    pose.orientation.z = 0.9603607654571533
    pose.orientation.w = 0.0

    # setPositions() parameter
    fraction_max_velocity = 0.2
    # positionInterpolations parameter
    execution_time = [5.0]
    # position only: True, axisMasks=7; False, axisMasks=63
    position_only = True

    rospy.wait_for_service('move_serviceServer')
    try:
        move_serviceServer = rospy.ServiceProxy('move_serviceServer', MoveJoints)
        return move_serviceServer(end_effector, frame,  pose, fraction_max_velocity, execution_time, position_only)
    except rospy.ServiceException as e:
        print("MoveClient: Service call failed: %s"%e)

def aruco_ex2():
    # Initializes aruco marker detection Node
    aruco_marker_node = Nodo()
    aruco_marker_node.start()

def ex2_2():
    aruco_marker_node = Nodo()
    aruco_marker_node.CameraBottom_optical_to_CameraBottom()
    print("MoveClient: Homogeneous transformation of CameraBottom_optical to CameraBottom: ", aruco_marker_node.H_camerabottomoptical_camerabottom)
 
if __name__ == "__main__": 
    if sys.argv[1] == "Ex1":
        print("MoveClient: Ex_1")
        result = move_ex1()
        print("MoveClient: Goal cartesian position:", result)
    elif sys.argv[1] == "Ex2":
        print("MoveClient: Ex_2")
        aruco_ex2()
    elif sys.argv[1] == "Ex2.2":
        print("MoveClient: Ex_2.2")
        ex2_2()
#!/usr/bin/env python3
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import rospy
from cv_bridge import CvBridge
import os
import numpy as np 
import pdb
import math
import tf
import tf2_ros
import motion
from naoqi import ALProxy
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from std_msgs.msg import Header


class Ball_Detector:
    def __init__(self):
        self.image = None
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(1)
        
        # Publisher ball pose
        self.pub_ball_pose = rospy.Publisher('/ball_pose_raw', Pose, queue_size=10)

        # Subscribe Bottom image
        self.image_sub = rospy.Subscriber("/nao_robot/camera/bottom/camera/image_raw", Image, self.image_callback) 

        # Subscribe Torso and Bottom camera pose
        self.torso_pose_sub = rospy.Subscriber('/torso_pose', PoseStamped, self.image_callback, queue_size=10)
        self.cam_bottom_pose_sub = rospy.Subscriber("/CameraBottom_frame", PoseStamped, self.image_callback, queue_size=10)
        
        self.current_image = None
        

    def image_callback(self, msg):
        rospy.loginfo('Image received...')
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding ="bgr8")
        self.current_image = image
        cv2.imshow("Bottom image", self.current_image)
        image_hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        
        ## Red, Blue, Green
        lower_red_0 = np.array([0, 90, 20])
        upper_red_0 = np.array([70, 255, 255])
        
        red2 = cv2.inRange(image_hsv, lower_red_0, upper_red_0)

        ## Dilate and Erode
        kernel = np.ones((5, 5), np.uint8)
        blue_dilate = cv2.dilate(red2, kernel, iterations=2)

        ## Blob detection
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Color
        params.filterByColor = True
        params.blobColor = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 200
        params.maxArea = 5000

        # Detect blobs.
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(blue_dilate)
        print("keypoints: ", keypoints)

        # Draw detected blobs as red circles
        im_with_keypoints = cv2.drawKeypoints(blue_dilate, keypoints, np.array([]), (36,255,12), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        cv2.imshow("Blob extraction", im_with_keypoints)      

        cv2.waitKey(10)
        

        ## Publish the ball position in the image 

        if keypoints: 
            
            # Calucalte ball pose from image data
            ball_center_x = int(keypoints[0].pt[0])
            print("ball_center_x:", ball_center_x)
            ball_center_y = int(keypoints[0].pt[1])
            print("ball_center_y:", ball_center_y)
            
            # Publish the ball pose 
            ball_pose_msg = Pose()
            ball_pose_msg.position.x = ball_center_x 
            ball_pose_msg.position.y = ball_center_y
            ball_pose_msg.position.z = 0
            ball_pose_msg.orientation.x = 0
            ball_pose_msg.orientation.y = 0
            ball_pose_msg.orientation.z = 0
            ball_pose_msg.orientation.w = 1
            self.pub_ball_pose.publish(ball_pose_msg)


    def start(self):   
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.init_node("ball_pose", anonymous=True)
        ball_detector = Ball_Detector()
        ball_detector.start()
    except rospy.ROSInterruptException:
        pass
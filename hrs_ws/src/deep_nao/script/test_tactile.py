from naoqi import ALProxy

if __name__ == '__main__':

    robotIP = "10.152.246.123"
    PORT = 9559
    
    # posture and motion proxy
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    motionProxy = ALProxy("ALMotion", robotIP, PORT)
    
    postureProxy.goToPosture("StandInit", 1.0)
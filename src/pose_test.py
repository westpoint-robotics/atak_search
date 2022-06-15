import rospy
from geometry_msgs.msg import PoseArray

def pose_cb(msg):
    pass



if __name__ == '__main__':
    rospy.init_node('test_sub')
    rospy.Subscriber('atak_search_wps',PoseArray, pose_cb)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        #rospy.loginfo('Waiting for msg')
        rate.sleep()
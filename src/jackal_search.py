"""
    AUthor: Jason Hughes
    Date: June 2022
    Project: DCIST CDE-A -- June Experiments
    ROS Node to take input from ATAK and provide waypoints to the ground robots 
    to search a specified area around a (not necesarily) specified object(s). 
"""

import rospy 
import cv2
import numpy as np
import tf2_ros as tf2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray, Pose, TransformStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from shapely.geometry import Point, Polygon, MultiPoint
from numpy import array
from cv_bridge import CvBridge


class JackalGoTo():

    def __init__(self):

        self.area_points = list()
        self.path_msg = Path()
        self.sem_img_msg = Image()
        self.center_msg = Point()
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer)

        self.img_center = tuple()
        self.map_recv = False
        self.img_center_ts = 0.0
        self.sem_map_ts = 0.0
        self.bridge = CvBridge()
        self.img_data = None
        self.viz_img = None
        self.img_height = 0.0
        self.img_width = 0.0

        self.img_scale = rospy.get_param('goto/img_scale')
        self.robot_list = rospy.get_param('goto/robot_names')
        self.tol = rospy.get_param('goto/tolerance')
        self.class_dict = rospy.get_param('goto/image_key')
        self.search_class = rospy.get_param('goto/default_class')
        self.base_frame= rospy.get_param('goto/base_frame')
        self.search_iter = self.class_dict[self.search_class]
        self.wp_proximity = rospy.get_param('goto/wp_proximity')

        rospy.Subscriber('Eros/atak/atak_path', Path, self.path_cb)
        rospy.Subscriber('Eros/atak/search_object', String, self.object_cb)
        rospy.Subscriber('asoom/map_sem_img', Image, self.sem_img_cb) # may need to be namespaced with quad
        rospy.Subscriber('asoom/map_sem_img_viz', Image, self.viz_cb)
        rospy.Subscriber('asoom/map_sem_img_center', PointStamped, self.img_center_cb)
        self.wps_pub = rospy.Publisher('atak_search_wps', PoseArray, queue_size = 10 )
        self.poly_pub = rospy.Publisher('poly_wps', PoseArray, queue_size = 10)

    def path_cb(self, msg):
        self.area_points.clear()
        self.path_msg = msg
        for pose in msg.poses:
            self.area_points.append((pose.pose.position.x, pose.pose.position.y)) # list of tuples: [(x,y),..]


    def sem_img_cb(self, msg):
        self.map_recv = True
        self.sem_img_msg = msg
        self.img_height = msg.height
        self.img_width = msg.width
        self.sem_map_ts = msg.header.stamp
        self.img_data = self.bridge.imgmsg_to_cv2(msg,"mono8")

    def viz_cb(self, msg):
        self.viz_img = self.bridge.imgmsg_to_cv2(msg,"bgr8")

        
    def img_center_cb(self, msg):
        self.center_msg = msg
        self.img_center_ts = msg.header.stamp
        self.img_center = (msg.point.x, msg.point.y)

    def object_cb(self, msg):
        if msg.data.lower() in self.class_dict.keys():
            self.search_class = msg.data
            self.search_iter = self.class_dict[msg.data]


    def check_timestamps(self):
        return self.img_center_ts == self.sem_map_ts

    def clip_image(self):
        #center_utm = self.tf_buffer.transform(self.center_msg, 'utm')
        (center_utm_e, center_utm_n) = 482916.06, 4421324.09 #center_utm.transform.translation.x, center_utm.transform.translation.y
        
        pixel_center = (self.img_width/2, self.img_height/2)
        
        manhattan_dist = []
        for point in self.area_points:
            manhattan_dist.append((round(center_utm_n-point[1]),round(center_utm_e-point[0]))) # list of tuple [(x_dist, y_dist)]
        
        pixel_coords = []
        for md in manhattan_dist:
            pixel_coords.append((pixel_center[0]+(md[0]/self.img_scale),pixel_center[1]+(md[1]/self.img_scale)))
        
        pixel_poly = Polygon(pixel_coords)
        self.polypub(pixel_coords)

        if not pixel_poly.is_valid:
            rospy.logwarn('The Search area from ATAK does not provide a valid search polygon')
            return None, None, (None,None), False
        
        max_y = int(max(pixel_coords, key=lambda x: x[0])[0])
        max_x = int(max(pixel_coords, key=lambda y: y[1])[1])
        min_y = int(min(pixel_coords, key=lambda x: x[0])[0])
        min_x = int(min(pixel_coords, key=lambda y: y[1])[1])
        

        if min([max_x,max_y,min_x,min_y]) < 0 or max_x > self.img_width or max_y > self.img_height:
            rospy.logwarn('One or more of the path vertices is outside the semantic map. Send a new area')
            return None, None, (None,None), False

        clipped_img = self.img_data[min_x:max_x,min_y:max_y]

        cropped_viz = self.viz_img[min_x:max_x,min_y:max_y]
        cv2.imwrite('/home/jason/Pictures/croppedImage.jpg', cropped_viz)

        return pixel_poly, clipped_img, (min_x, min_y), True

    def generate_wps(self, poly, img, top_left):
        height, width = img.shape[0], img.shape[1]
        
        pixel_wps = []
        iter = 0 
        safe_list = [self.class_dict['road'], self.class_dict['grass'], self.class_dict['gravel']]

        while iter < height:
            # is there red?
            row = img[int(iter),:]
            elem = np.sum(row == self.search_iter)
            if elem > 2:

                if np.any(np.isin(row,safe_list)): # if there is road grass or gravel
                    # make a wp
                    where_obj_class = np.where(row == self.class_dict[self.search_class])[0]
                    where_safe_class =  np.sort(np.concatenate((np.where(row==safe_list[2])[0],np.concatenate((np.where(row==safe_list[0])[0],np.where(row==safe_list[1])[0]),axis=None)), axis=None),axis=None)
                    
                    for index in where_obj_class:
                        #check if there are safe pixels within a threshold of the building
                        safe_close = np.isclose(where_safe_class, index, atol=self.tol/self.img_scale)
                        
                        if safe_close.any(): #if there are safe wps near the buildings ...
                            goto_x_index = np.where(safe_close)[0] # safe pixels within the threshold: array([i0 i1 i2 ...])
                            if len(goto_x_index) > 1:
                                diffs = np.diff(goto_x_index)
                                split = np.argmax(diffs)
                                x_val = 0
                                if split != 0:
                                    stopped = 0

                                    for i in range(len((diffs))):
                                        if diffs[i] > 10:
                                            x_ind = goto_x_index[stopped:i]
                                            #print('1: ',x_ind)
                                            x_ind = x_ind[len(x_ind)//2]
                                            #print('2: ',x_ind)
                                            x_val = where_safe_class[x_ind]
                                            #print('3: ', x_val)
                                            stopped = i
                                else:
                                    x_ind = goto_x_index[len(goto_x_index)//2]
                                    x_val = where_safe_class[x_ind]
                                pixel_wps.append((x_val,int(iter)))
                            else:
                                pass
                    iter += self.wp_proximity/self.img_scale #wp closeness threshold -- NEEDS TO BE PARAMETERIZED
                else:
                    iter += 1 
            else:
                iter += 1     
        
        clean_wps = self.clean_pixels(pixel_wps)
        full_frame_wps = self.get_full_pic_coords(clean_wps, top_left)
        tested_wps = self.test_wps(poly, full_frame_wps)

        local_wps = list()
        for wp in tested_wps:
            local_wps.append(self.convert_to_world(wp))

        return local_wps

    def clean_pixels(self, pixels):
        dirty_arr = np.array(pixels)
        clean_arr = list()
        clean_arr.append(pixels[0])
        x_changes = np.where(np.diff(dirty_arr[:,1]) != 0)

        for c in x_changes[0]:
            clean_arr.append(pixels[c])

        return clean_arr

    def get_full_pic_coords(self, wps, top_left):
        # take wps in the cropped image frame and transform them to full image coords
        return [(wp[0]+top_left[1],wp[1]+top_left[0]) for wp in wps]


    def test_wps(self, poly, wps):
        tested_wps = list()
        
        for wp in wps:
            p = Point(wp)
            if poly.contains(p):
                tested_wps.append(wp)
        return tested_wps


    def convert_to_world(self, pt):
        #pixel coordinate to robots world frame
        img_pt = np.array(pt) - np.array((self.img_width/2,self.img_height/2))
        img_pt = img_pt*self.img_scale
        world_pt = -img_pt
        world_pt = world_pt + (self.img_center[1],self.img_center[0])
        return (world_pt[1],world_pt[0])
        

    def world_to_img(self,pt):
        world_pt = np.array(pt)-self.img_center
        img_pt = np.array([-world_pt[1],-world_pt[0]])
        img_pt = img_pt * self.img_scale
        img_pt = img_pt + np.array([self.img_height/2,self.img_width/2])
        

    def publish(self, wps):
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        pose_list = list()
        for wp in wps:
            p = Pose()
            p.position.x = wp[0]
            p.position.y = wp[1]
            pose_list.append(p)
        msg.poses = (pose_list)
        self.wps_pub.publish(msg)

    def polypub(self,wps):
        msg=PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        pose_list = list()
        for wp in wps:
            lwp = self.convert_to_world(wp)
            p = Pose()
            p.position.x = lwp[0]
            p.position.y = lwp[1]
            pose_list.append(p)
        msg.poses = (pose_list)
        
        self.poly_pub.publish(msg)


    def cycle(self):

        try:
            rate = rospy.Rate(1)

            while not rospy.is_shutdown():
                
                if self.map_recv:
                    if len(self.area_points) != 0:
                        if self.check_timestamps:
                            pixel_poly, clipped_img, mins, valid_path = self.clip_image() # returns shapely Polygon, cv2 image, top_left pixel where crop began, continue bool
                            if valid_path:
                                local_frame_wps = self.generate_wps(pixel_poly, clipped_img, mins)
                                self.publish(local_frame_wps)
                                self.area_points.clear()
                        else:
                            rospy.loginfo('Waiting for Map_Center and Semantic_Map timestamps to align')
                    else:
                        rospy.loginfo('Waiting for a search area...')
                rate.sleep()
        except rospy.ROSInterruptException:
            pass

            
if __name__ == '__main__':
     
    rospy.init_node('atak_goto', anonymous=True)
    goto = JackalGoTo()
    goto.cycle()


# TODO:
# 1.  Search image for building edges -- DONE
# 2.  Generate wps based on search and convert them from pixel -> local -- DONE
# 3.  Check if waypoints make sense -- DONE
# 4.  Add resiliency, i.e. -- DONE
#     - polygon is valid, -- DONE
#     - all points on path are within the semantic map, -- DONE
#     - timestamp for center and image align -- DONE
#     - recieved a map -- DONE
# 5.  Check if I'm mixing up x's and y's when cropping image -- DONE
# 6.  Publish the local waypoints -- DONE
# 7.  Don't generate wps near each other -- ONGOING
#     - make this a parameter -- DONE
# 8.  Ability to look around cars too -- DONE
# 9.  Add in go_to message from ATAK -- NOT STARTED 
# 10. Code clean up -- ONGOING
#     - no print outs -- DONE
#     - logwarn for invalid inputs -- DONE
#     - unsubscribe to the viz topic, don't crop the image -- keep for now
#     - no need for polypub -- keep for now
# 11. Make a Repo
        

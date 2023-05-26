#!/usr/bin/env python3

# from pure_pursuit.module_to_import import *
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from scipy.interpolate import splev, splprep

from scipy.spatial.transform import Rotation as R


# TODO: import ROS msg types and libraries

class PurePursuit(Node):
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        filename = "sim_points.csv" 

        self.steer_lookahead = 1.5
        self.kp = 0.25
        self.pose_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 1)
        
        
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, 'drive',1)
        self.goal_points_publisher = self.create_publisher(MarkerArray, 'pp_goal_points',1)
        self.spline_publisher = self.create_publisher(Marker, 'pp_spline',1)
        self.pp_goal_publisher = self.create_publisher(Marker, 'pp_goal_point',1)
        
        self.goal_points = load_points(filename, scaler=1)

        spline_data, _ = splprep(self.goal_points.T, s=0, per=True)
        self.x_spline, self.y_spline = splev(np.linspace(0, 1, 1000), spline_data)
        # print(len(self.x_spline))
        self.pp_points_data = self.visualize_pp_goal_points()
        self.pp_spline_data = self.visualize_spline()
        #Publish Rviz Markers every 2 seconds
        self.timer = self.create_timer(2, self.publish_rviz_data)#Publish waypoints



    def pose_callback(self, pose_msg):
        # Find the current waypoint to track using methods mentioned in lecture
        current_position = pose_msg.pose.pose.position
        current_quat = pose_msg.pose.pose.orientation

        quaternion = [current_quat.x, current_quat.y, current_quat.z, current_quat.w]
        global_car_position = [current_position.x, current_position.y, current_position.z] # Current location of car in world frame

        # Calculate immediate goal on Pure Pursuit Trajectory
        spline_points = np.hstack((self.x_spline.reshape(-1,1), self.y_spline.reshape(-1,1), np.zeros_like(self.y_spline.reshape(-1,1))))
        
        # Calculate closest point on spline
        norm_array = np.linalg.norm(spline_points - global_car_position, axis = 1)
        closest_pt_idx = np.argmin(norm_array)

        # Check if car is oriented opposite the spline array direction
        if(closest_pt_idx+10>(len(self.x_spline)-1)): idx = 10 
        else: idx =  closest_pt_idx+10
        sample_point = global_2_local(quaternion, spline_points[idx], global_car_position)
        if sample_point[0]>0:
            arangeit = np.arange(len(self.x_spline))
            rollit = np.roll(arangeit, -closest_pt_idx)
        else:
            arangeit = np.flip(np.arange(len(self.x_spline)))
            rollit = np.roll(arangeit, closest_pt_idx)

        # Find the point on spline that is 1 lookahead away.
        cumsum_spline = (np.cumsum(np.linalg.norm(spline_points[rollit]-global_car_position, axis = 1) > self.steer_lookahead))    
        lookahead_idx = np.where(cumsum_spline>0)
        pp_goal_point_global = spline_points[rollit[lookahead_idx[0][0]]] # Current point on spline to follow 
        self.visualize_pt(pp_goal_point_global)

        # Calculate curvature/steering angle
        pp_goal_point_local = global_2_local(quaternion, pp_goal_point_global, global_car_position)
        steering_angle = self.calc_steer(pp_goal_point_local, self.kp)
        print("steering_angle: ",steering_angle)

        msg = AckermannDriveStamped()
        msg.drive.speed = 1.0 #float(drive_speed)###CHANGE THIS BACK, IN SIM THE CHANGING VELOCITY WAS CAUSING PROBLEMS
        msg.drive.steering_angle = float(steering_angle)
        self.drive_publisher.publish(msg)
        # TODO: publish drive message, don't forget to limit the steering angle between -0.4189 and 0.4189 radians
    
    def calc_steer(self, goal_point_car, kp):
        """
        Returns the steering angle from the local goal point
        """
        y = goal_point_car[1]
        steer_dir = np.sign(y)
        r = self.steer_lookahead ** 2 / (2 * np.abs(y))
        gamma = 1 / r
        steering_angle = (gamma * kp * steer_dir)
        return steering_angle

    ########################### VISUALIZATION ############################
    def visualize_pt(self, point):
        array_values=MarkerArray()

        message = Marker()
        message.header.frame_id="map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.type= Marker.SPHERE
        message.action = Marker.ADD
        message.id=0
        message.pose.orientation.x=0.0
        message.pose.orientation.y=0.0
        message.pose.orientation.z=0.0
        message.pose.orientation.w=1.0
        message.scale.x=0.2
        message.scale.y=0.2
        message.scale.z=0.2
        message.color.a=1.0
        message.color.r=1.0
        message.color.b=1.0
        message.color.g=0.0
        message.pose.position.x=float(point[0])
        message.pose.position.y=float(point[1])
        message.pose.position.z=0.0
        array_values.markers.append(message)
        self.pp_goal_publisher.publish(message)
    
    def visualize_pp_goal_points(self):
        array_values=MarkerArray()

        for i in range(len(self.goal_points)):
            message = Marker()
            message.header.frame_id="map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.action = Marker.ADD
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.2
            message.scale.y=0.2
            message.scale.z=0.2
            message.color.a=1.0
            message.color.r=1.0
            message.color.b=0.0
            message.color.g=0.0
            message.pose.position.x=float(self.goal_points[i,0])
            message.pose.position.y=float(self.goal_points[i,1])
            message.pose.position.z=0.0
            array_values.markers.append(message)
        return array_values
    
    def visualize_spline(self):

        message = Marker()
        message.header.frame_id="map"
        message.type= Marker.LINE_STRIP
        message.action = Marker.ADD
        message.pose.position.x= 0.0
        message.pose.position.y= 0.0
        message.pose.position.z=0.0
        message.pose.orientation.x=0.0
        message.pose.orientation.y=0.0
        message.pose.orientation.z=0.0
        message.pose.orientation.w=1.0
        message.scale.x=0.1

        message.color.a=1.0
        message.color.r=0.0
        message.color.b=1.0
        message.color.g=1.0

        for i in range(len(self.x_spline)-1):
            message.id=i
            message.header.stamp = self.get_clock().now().to_msg()

            point1 = Point()
            point1.x = float(self.x_spline[i])
            point1.y = float(self.y_spline[i])
            point1.z = 0.0
            message.points.append(point1)

            point2 = Point()
            point2.x = float(self.x_spline[i+1])
            point2.y = float(self.y_spline[i+1])
            point2.z = 0.0
            message.points.append(point2)
            self.spline_publisher.publish(message)

        return message
    
    def publish_rviz_data(self):
        self.goal_points_publisher.publish(self.pp_points_data)
        self.spline_publisher.publish(self.pp_spline_data)
    ########################### VISUALIZATION ############################
    
def global_2_local(quaternion, pt_w, T_c_w):
    # Transform goal point to vehicle frame of reference
    rot = (R.as_matrix(R.from_quat(quaternion)))
    pt_c = (np.array(pt_w) - np.array(T_c_w))@rot
    """ 
    # Alternate Method 
    H_global2car = np.zeros([4, 4]) #rigid body transformation from  the global frame of referce to the car
    H_global2car[3, 3] = 1
    current_rotation_matrix = R.from_quat(np.array([current_quat.x,current_quat.y,current_quat.z,current_quat.w])).as_matrix()
    H_global2car[0:3, 0:3] = np.array(current_rotation_matrix)
    H_global2car[0:3, 3] = np.array([current_position.x, current_position.y, current_position.z])

    # Calculate point
    goal_point_global = np.append(pt_w, 1).reshape(4, 1)
    pt_c = np.linalg.inv(H_global2car) @ goal_point_global
    """
    return pt_c

def load_points(file, scaler=10):
    # Open csv and read the waypoint data
    with open(file, 'r') as f:
        lines = (line for line in f if not line.startswith('#'))
        data = np.loadtxt(lines, delimiter=',', dtype=float)
    points = data / scaler

    return points

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
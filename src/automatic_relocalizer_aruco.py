#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import time
import numpy as np
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
import tf2_ros
import tf_transformations
import yaml
import os

MARKER_LENGTH = 0.060
OPENCV_TO_ROS = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0]
])

BACKUP_OFFSETS = {
    'left':  [0.00, -0.80, 0.00],
    'right': [0.00,  0.80, 0.00],
    'front': [0.00,  0.00, 0.00],
    'back':  [0.00,  0.00, 0.00],
    'none':  [0.00,  0.00, 0.00],
}

AMCL_COV_X_THRESH = 0.03
AMCL_COV_Y_THRESH = 0.03
AMCL_COV_YAW_THRESH = 0.08
IMAGE_AVAIL_TIMEOUT = 3.0

ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters_create()

def quaternion_from_yaw(yaw):
    q = tf_transformations.quaternion_from_euler(0, 0, yaw)
    quat = Quaternion()
    quat.x, quat.y, quat.z, quat.w = q
    return quat

def orthonormalize_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt

def rotation_matrix_to_quaternion_np(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def load_marker_config(file_path):
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to open marker YAML '{file_path}': {e}")

    markers = data.get('markers', {})
    marker_poses, marker_sides = {}, {}

    for mid, info in markers.items():
        mid_int = int(mid)
        pose_raw = info.get('pose', [0.0, 0.0, 0.0])
        mx, my, myaw = map(float, pose_raw)
        marker_poses[mid_int] = [mx, my, myaw]
        marker_sides[mid_int] = str(info.get('side', 'none'))

    return marker_poses, marker_sides

class ArucoDriftRelocalizer(Node):
    def __init__(self):
        super().__init__('aruco_drift_relocalizer_service')

        # Parameters
        self.declare_parameter('marker_yaml', '/home/user/config.yaml')
        self.declare_parameter('monitor_period', 2.0)
        self.declare_parameter('relocalize_cooldown', 10.0)
        self.declare_parameter('drift_trigger_score', 0.7)

        yaml_path = self.get_parameter('marker_yaml').get_parameter_value().string_value
        if not os.path.isfile(yaml_path):
            self.get_logger().error(f"Marker YAML not found: {yaml_path}")
            self.marker_map_poses, self.marker_side = {}, {}
        else:
            self.marker_map_poses, self.marker_side = load_marker_config(yaml_path)
            self.get_logger().info(f"Loaded {len(self.marker_map_poses)} markers from {yaml_path}")

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Camera / image
        self.camera_matrix, self.dist_coeffs, self.latest_image = None, None, None
        self.bridge = CvBridge()

        # AMCL covariance monitor
        self.latest_amcl_cov = None

        # Subscriptions
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_cb, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)

        # Publisher + service
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.srv = self.create_service(Trigger, '/relocalize_once', self.service_relocalize_cb)

        # Automatic drift monitor
        self.last_relocalize_time = 0.0
        self.monitor_period = self.get_parameter('monitor_period').get_parameter_value().double_value
        self.relocalize_cooldown = self.get_parameter('relocalize_cooldown').get_parameter_value().double_value
        self.drift_trigger_score = self.get_parameter('drift_trigger_score').get_parameter_value().double_value
        self.create_timer(self.monitor_period, self._drift_monitor_cb)

        # Initial relocalization trigger
        self.initial_relocalize_done = False
        self.create_timer(2.0, self._initial_relocalize_cb)

        self.get_logger().info("Aruco drift relocalizer started (auto + initial relocalization enabled).")

    def _initial_relocalize_cb(self):
        if self.initial_relocalize_done:
            return

        success, msg = self._do_relocalize()
        if success:
            self.get_logger().info(f"Initial relocalization succeeded: {msg}")
            self.initial_relocalize_done = True
        else:
            self.get_logger().warn(f"Initial relocalization retrying... ({msg})")

    def camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("CameraInfo received.")

    def image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")

    def amcl_cb(self, msg: PoseWithCovarianceStamped):
        cov = msg.pose.covariance
        try:
            self.latest_amcl_cov = {
                'cov_x': float(cov[0]),
                'cov_y': float(cov[7]),
                'cov_yaw': float(cov[35]),
            }
        except Exception:
            self.latest_amcl_cov = None

    def service_relocalize_cb(self, request, response):
        success, message = self._do_relocalize()
        response.success = bool(success)
        response.message = str(message)
        return response

    def _drift_monitor_cb(self):
        if self.latest_amcl_cov is None:
            return

        now = time.time()
        if now - self.last_relocalize_time < self.relocalize_cooldown:
            return

        c = self.latest_amcl_cov
        sx = c['cov_x'] / AMCL_COV_X_THRESH
        sy = c['cov_y'] / AMCL_COV_Y_THRESH
        syaw = c['cov_yaw'] / AMCL_COV_YAW_THRESH
        score = (sx + sy + syaw) / 3.0

        if score >= self.drift_trigger_score:
            self.get_logger().warn(f"AMCL drift detected (score={score:.3f}) → auto relocalizing")
            success, msg = self._do_relocalize()
            if success:
                self.get_logger().info("Automatic relocalization succeeded.")
            else:
                self.get_logger().warn(f"Automatic relocalization failed: {msg}")
            self.last_relocalize_time = now

    def _do_relocalize(self):
        if not self.marker_map_poses:
            return False, "No marker map loaded."
        if self.camera_matrix is None:
            return False, "CameraInfo not received."
        if self.latest_image is None:
            return False, "No camera image available."

        img = self.latest_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        if ids is None:
            return False, "No ArUco markers detected."

        ids = ids.flatten()
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, self.camera_matrix, self.dist_coeffs)

        visible_markers = []
        best_idx, best_id, best_dist, best_rvec, best_tvec = None, None, 1e9, None, None
        for i in range(len(ids)):
            tvec_cv = tvecs[i][0].reshape(3, 1)
            tvec_ros = OPENCV_TO_ROS @ tvec_cv.flatten()
            dist = np.linalg.norm(tvec_ros)
            visible_markers.append((int(ids[i]), dist))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_id = int(ids[i])
                best_rvec = rvecs[i]
                best_tvec = tvecs[i].reshape(3, 1)

        if len(visible_markers) > 1:
            vis_info = ", ".join([f"id={mid} ({d:.2f}m)" for mid, d in visible_markers])
            self.get_logger().info(f"Multiple markers visible → {vis_info}")

        if best_idx is None:
            return False, "No valid marker selected."
        self.get_logger().info(f"Closest marker selected: ID={best_id} at {best_dist:.3f} m")

        if best_id not in self.marker_map_poses:
            return False, f"Marker ID {best_id} not in marker map."

        try:
            mx, my, myaw = map(float, self.marker_map_poses[best_id])
            R_map_marker = np.array([
                [math.cos(myaw), -math.sin(myaw), 0],
                [math.sin(myaw),  math.cos(myaw), 0],
                [0, 0, 1]
            ])
            T_map_marker = np.eye(4)
            T_map_marker[:3, :3] = R_map_marker
            T_map_marker[:3, 3] = [mx, my, 0.0]

            R_cv, _ = cv2.Rodrigues(best_rvec)
            R_ros = OPENCV_TO_ROS @ R_cv @ OPENCV_TO_ROS.T
            t_ros = (OPENCV_TO_ROS @ best_tvec.flatten()).reshape(3, 1)
            T_marker_camera = np.eye(4)
            T_marker_camera[:3, :3] = R_ros
            T_marker_camera[:3, 3] = t_ros.flatten()

            T_camera_base = np.eye(4)
            T_map_robot = T_map_marker @ np.linalg.inv(T_marker_camera) @ T_camera_base

            rx, ry, rz = T_map_robot[:3, 3]
            R_robot = orthonormalize_rotation(T_map_robot[:3, :3])
            yaw = math.atan2(R_robot[1, 0], R_robot[0, 0])

            side = str(self.marker_side.get(best_id, 'none'))
            bx, by, byaw = BACKUP_OFFSETS.get(side, BACKUP_OFFSETS['none'])
            bx, by, byaw = map(float, [bx, by, byaw])

            if side == 'left':
                ry += by
            elif side == 'right':
                ry += by
            elif side == 'front':
                rx += bx
            elif side == 'back':
                rx += bx
            yaw += byaw

            pose = PoseWithCovarianceStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.pose.position.x = float(rx)
            pose.pose.pose.position.y = float(ry)
            pose.pose.pose.orientation = quaternion_from_yaw(yaw)
            pose.pose.covariance = [0.0] * 36
            pose.pose.covariance[0] = 0.25
            pose.pose.covariance[7] = 0.25
            pose.pose.covariance[35] = 0.0685

            self.pose_pub.publish(pose)
            rclpy.spin_once(self, timeout_sec=0.01)
            self.pose_pub.publish(pose)

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'relocalized_robot_camera'
            t.transform.translation.x = float(rx)
            t.transform.translation.y = float(ry)
            t.transform.translation.z = 0.0
            q = rotation_matrix_to_quaternion_np(R_robot)
            t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = q
            self.tf_broadcaster.sendTransform(t)

            msg = f"/initialpose published from marker {best_id} side={side} rx={rx:.3f} ry={ry:.3f} yaw={yaw:.3f}"
            return True, msg

        except Exception as e:
            return False, f"Exception during pose computation: {e}"

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDriftRelocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

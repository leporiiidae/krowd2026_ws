"""Detect and persistently track left/right cone lines from Ouster clouds."""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .algorithms import cone_centroids, quaternion_matrix, voxel_downsample


Point2D = tuple[float, float]
ChainPoint = tuple[float, float, int]
Pose2D = tuple[float, float, float]
TimedPose = tuple[float, float, float, float]


class ConePerceptionNode(Node):
    """PointCloud2 equivalent of xycar_ws's scan-based cone perception."""

    def __init__(self) -> None:
        super().__init__("stop_mission_cone_perception")
        defaults = {
            "pointcloud_topic": "/ouster/points",
            "odometry_topic": "/ekf/odometry",
            "start_topic": "/stop_mission/start",
            "auto_start": True,
            "frame_id": "base_link",
            "marker_topic": "/stop_mission/cones",
            "raw_marker_topic": "/stop_mission/raw_cones",
            "chain_marker_topic": "/stop_mission/chain_markers",
            "sector_marker_topic": "/stop_mission/search_sectors",
            "virtual_marker_topic": "/stop_mission/virtual_cones",
            "status_topic": "/stop_mission/perception_status",
            "min_x": 0.2,
            "max_x": 12.0,
            "max_abs_y": 8.0,
            "min_z": 0.05,
            "max_z": 0.90,
            "voxel_size": 0.05,
            "cluster_tolerance": 0.28,
            "min_cluster_points": 3,
            "max_cluster_points": 180,
            "max_cluster_diameter": 0.65,
            "min_cluster_height": 0.08,
            "max_cluster_top_z": 0.50,
            "seed_min_x": 1.5,
            "seed_min_abs_y": 1.0,
            "side_match_max_distance": 5.0,
            "side_sector_angle_deg": 35.0,
            "side_sector_radius": 5.0,
            "initial_chain_min_length": 2,
            "side_chain_duplicate_distance": 0.55,
            "tracking_match_distance": 0.80,
            "tracking_min_classified": 2,
            "tracking_reinitialize_frames": 2,
            "sector_tail_size": 2,
            "max_chain_points": 40,
            "chain_prune_behind_x": -0.5,
            "motion_min_dt": 0.005,
            "yaw_jump_threshold_deg": 5.0,
            "yaw_stable_threshold_deg": 1.0,
            "yaw_stable_frames": 5,
            "odometry_buffer_size": 500,
            "time_rewind_tolerance_sec": 0.1,
            "enable_virtual_cones": False,
            "virtual_track_width": 6.0,
            "virtual_count_diff_threshold": 3,
            "virtual_max_count": 8,
            "tf_timeout": 0.05,
            "cone_diameter": 0.30,
            "cone_height": 0.65,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.started = bool(self.parameter("auto_start"))
        self.left_chain: list[ChainPoint] = []
        self.right_chain: list[ChainPoint] = []
        self.virtual_left: list[Point2D] = []
        self.virtual_right: list[Point2D] = []
        self.last_cloud_time: float | None = None
        self.previous_motion_time: float | None = None
        self.previous_motion_pose: Pose2D | None = None
        self.last_odometry_time: float | None = None
        self.initialized = False
        self.tracking_loss_count = 0
        self.yaw_realigning = False
        self.yaw_stable_count = 0
        self.last_motion_debug = "motion wait"
        self.odometry_buffer: deque[TimedPose] = deque(
            maxlen=int(self.parameter("odometry_buffer_size"))
        )
        self.data_active = False
        self.published_marker_counts = {
            "classified": {"blue": 0, "yellow": 0, "raw": 0},
            "raw": {"blue": 0, "yellow": 0, "raw": 0},
            "chain": {"blue": 0, "yellow": 0, "raw": 0},
            "virtual": {"blue": 0, "yellow": 0, "raw": 0},
        }
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(
            PointCloud2,
            str(self.parameter("pointcloud_topic")),
            self.cloud_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool, str(self.parameter("start_topic")), self.start_callback, 10
        )
        self.create_subscription(
            Odometry,
            str(self.parameter("odometry_topic")),
            self.odometry_callback,
            qos_profile_sensor_data,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.parameter("marker_topic")), 10
        )
        self.raw_marker_pub = self.create_publisher(
            MarkerArray, str(self.parameter("raw_marker_topic")), 10
        )
        self.chain_marker_pub = self.create_publisher(
            MarkerArray, str(self.parameter("chain_marker_topic")), 10
        )
        self.sector_marker_pub = self.create_publisher(
            MarkerArray, str(self.parameter("sector_marker_topic")), 10
        )
        self.virtual_marker_pub = self.create_publisher(
            MarkerArray, str(self.parameter("virtual_marker_topic")), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.parameter("status_topic")), 10
        )
        self.get_logger().info(
            "stop_mission perception: %s -> %s (persistent compensated chains)"
            % (
                self.parameter("pointcloud_topic"),
                self.parameter("marker_topic"),
            )
        )
        self.get_logger().info(
            "Waiting for a transformable PointCloud2; cone classification starts "
            "when bag/live sensor data arrives"
        )

    def parameter(self, name: str):
        return self.get_parameter(name).value

    def start_callback(self, message: Bool) -> None:
        if not message.data:
            self.started = False
            self.reset_tracking()
            return
        if not self.started:
            self.reset_tracking()
        self.started = True

    def odometry_callback(self, message: Odometry) -> None:
        stamp = self.message_time(message.header.stamp)
        tolerance = float(self.parameter("time_rewind_tolerance_sec"))
        if (
            self.last_odometry_time is not None
            and stamp < self.last_odometry_time - tolerance
        ):
            self.get_logger().info(
                "Odometry time moved backwards; clearing EKF history for a new replay"
            )
            self.reset_tracking()
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = self.quaternion_to_yaw(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        self.odometry_buffer.append((
            stamp,
            float(position.x),
            float(position.y),
            yaw,
        ))
        self.last_odometry_time = stamp

    def reset_tracking(self) -> None:
        self.left_chain.clear()
        self.right_chain.clear()
        self.virtual_left.clear()
        self.virtual_right.clear()
        self.last_cloud_time = None
        self.previous_motion_time = None
        self.previous_motion_pose = None
        self.last_odometry_time = None
        self.initialized = False
        self.tracking_loss_count = 0
        self.yaw_realigning = False
        self.yaw_stable_count = 0
        self.odometry_buffer.clear()
        self.data_active = False

    def cloud_callback(self, message: PointCloud2) -> None:
        if not self.started:
            return
        cloud_time = self.message_time(message.header.stamp)
        tolerance = float(self.parameter("time_rewind_tolerance_sec"))
        if (
            self.last_cloud_time is not None
            and cloud_time < self.last_cloud_time - tolerance
        ):
            self.get_logger().info(
                "Point-cloud time moved backwards; resetting cone tracking for a new replay"
            )
            self.reset_tracking()
        lidar_reclassification = self.compensate_chains_to_time(cloud_time)
        self.age_chains()
        try:
            points = self.cloud_in_base_link(message)
        except (TransformException, ValueError) as error:
            self.get_logger().warn(f"Skipping cloud: {error}", throttle_duration_sec=2.0)
            return

        if not self.data_active:
            self.get_logger().info(
                "First valid point cloud received; starting cone classification"
            )
            self.data_active = True
        self.last_cloud_time = cloud_time

        roi = self.filter_roi(points)
        roi = voxel_downsample(roi, float(self.parameter("voxel_size")))
        cones = cone_centroids(
            roi,
            float(self.parameter("cluster_tolerance")),
            int(self.parameter("min_cluster_points")),
            int(self.parameter("max_cluster_points")),
            float(self.parameter("max_cluster_diameter")),
            float(self.parameter("min_cluster_height")),
            float(self.parameter("max_cluster_top_z")),
        )
        was_initialized = self.initialized
        if lidar_reclassification:
            # During an EKF heading realignment, rebuild from only the current
            # LiDAR centroids every frame. This deliberately avoids carrying a
            # coordinate-frame correction into the persistent cone geometry.
            labels = self.classify_initial(cones)
            self.tracking_loss_count = 0
        elif was_initialized:
            labels = self.classify_sector_only(cones)
            classified_count = labels.count("blue") + labels.count("yellow")
            if classified_count < int(self.parameter("tracking_min_classified")):
                self.tracking_loss_count += 1
            else:
                self.tracking_loss_count = 0

            reinitialize_frames = int(self.parameter("tracking_reinitialize_frames"))
            if (
                self.tracking_loss_count >= reinitialize_frames
                and self.can_reinitialize(cones)
            ):
                self.get_logger().warn(
                    "Cone-chain association was lost; rebuilding both sides from "
                    "the current LiDAR cone coordinates",
                    throttle_duration_sec=1.0,
                )
                self.initialized = False
                labels = self.classify_initial(cones)
                self.tracking_loss_count = 0
        else:
            labels = self.classify_initial(cones)
            self.tracking_loss_count = 0
        self.update_chains_from_labels(cones, labels)
        self.prune_chains()

        # `/stop_mission/cones` is both the RViz display and the planner input.
        # Publish only current-frame LiDAR centroids here: these are exactly the
        # gray-cylinder coordinates above, with a blue/yellow label applied.  The
        # EKF-compensated persistent chain remains an association/search aid and
        # must never leak its shifted historical coordinates into path planning.
        current_points = [
            point for point, label in zip(cones, labels)
            if label in ("blue", "yellow")
        ]
        current_labels = [label for label in labels if label in ("blue", "yellow")]
        # Mirror from the same measured coordinates that are published and planned
        # with, so a virtual cone never inherits an EKF-compensated chain offset.
        self.update_virtual_cones(
            [point for point, label in zip(cones, labels) if label == "blue"],
            [point for point, label in zip(cones, labels) if label == "yellow"],
        )

        stamp = message.header.stamp
        frame = str(self.parameter("frame_id"))
        self.raw_marker_pub.publish(
            self.marker_array(cones, ["raw"] * len(cones), frame, stamp, "raw")
        )
        self.marker_pub.publish(
            self.marker_array(
                current_points, current_labels, frame, stamp, "classified"
            )
        )

        chain_points = self.front_chain_points(self.left_chain)
        chain_labels = ["blue"] * len(chain_points)
        right_chain_points = self.front_chain_points(self.right_chain)
        chain_points.extend(right_chain_points)
        chain_labels.extend(["yellow"] * len(right_chain_points))
        self.chain_marker_pub.publish(
            self.marker_array(chain_points, chain_labels, frame, stamp, "chain")
        )
        self.sector_marker_pub.publish(self.sector_marker_array(frame, stamp))
        # Visualization only: the synthetic opposite-side cones never enter
        # `/stop_mission/cones`, so the planner keeps using its own one-side
        # offset fallback while RViz still shows what was inferred.
        self.virtual_marker_pub.publish(self.virtual_marker_array(frame, stamp))
        self.publish_status(len(points), len(roi), len(cones), labels)

    def cloud_in_base_link(self, message: PointCloud2) -> np.ndarray:
        source_frame = message.header.frame_id
        target_frame = str(self.parameter("frame_id"))
        if not source_frame:
            raise ValueError("PointCloud2 frame_id is empty")
        values = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        points = np.column_stack((values["x"], values["y"], values["z"])).astype(
            float, copy=False
        )
        if source_frame == target_frame:
            return points
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(message.header.stamp),
            timeout=Duration(seconds=float(self.parameter("tf_timeout"))),
        ).transform
        rotation = transform.rotation
        matrix = quaternion_matrix(
            rotation.x, rotation.y, rotation.z, rotation.w
        )
        translation = np.asarray([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ])
        return points @ matrix.T + translation

    def filter_roi(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        finite = np.all(np.isfinite(points), axis=1)
        mask = (
            finite
            & (points[:, 0] >= float(self.parameter("min_x")))
            & (points[:, 0] <= float(self.parameter("max_x")))
            & (np.abs(points[:, 1]) <= float(self.parameter("max_abs_y")))
            & (points[:, 2] >= float(self.parameter("min_z")))
            & (points[:, 2] <= float(self.parameter("max_z")))
        )
        return points[mask]

    def classify_initial(self, cones: Sequence[Point2D]) -> list[str]:
        labels = ["unknown"] * len(cones)
        left_index = self.closest_seed_candidate(cones, "blue")
        right_index = self.closest_seed_candidate(cones, "yellow")
        if left_index is not None:
            labels[left_index] = "blue"
        if right_index is not None:
            labels[right_index] = "yellow"

        changed = True
        while changed:
            changed = False
            for side in ("blue", "yellow"):
                side_points = [
                    point for point, label in zip(cones, labels) if label == side
                ]
                if len(side_points) != 1:
                    continue
                result = self.closest_unknown_to_origin(
                    cones,
                    labels,
                    side_points[0],
                    float(self.parameter("side_match_max_distance")),
                )
                if result is not None:
                    labels[result[0]] = side
                    changed = True

            for side in ("blue", "yellow"):
                temporary = [
                    point
                    for point, label in zip(cones, labels)
                    if label == side
                ]
                sector = self.sector(temporary)
                if sector is None:
                    continue
                origin, direction = sector
                for index, point in enumerate(cones):
                    if labels[index] != "unknown":
                        continue
                    if self.sector_score(origin, direction, point) is not None:
                        labels[index] = side
                        changed = True

        minimum = int(self.parameter("initial_chain_min_length"))
        left = [point for point, label in zip(cones, labels) if label == "blue"]
        right = [point for point, label in zip(cones, labels) if label == "yellow"]
        self.left_chain = self.points_to_chain(left)
        self.right_chain = self.points_to_chain(right)
        self.initialized = len(left) >= minimum and len(right) >= minimum
        return labels

    def classify_sector_only(self, cones: Sequence[Point2D]) -> list[str]:
        labels = ["unknown"] * len(cones)
        match_distance = float(self.parameter("tracking_match_distance"))

        # Re-observed cones keep the side of their nearest persistent-chain
        # point, but retain their current LiDAR coordinates for publication.
        for index, point in enumerate(cones):
            candidates: list[tuple[float, str]] = []
            for side, chain in (
                ("blue", self.left_chain),
                ("yellow", self.right_chain),
            ):
                closest = self.closest_chain_index(chain, point)
                if closest is None:
                    continue
                distance = math.dist(self.chain_xy(chain[closest]), point)
                if distance <= match_distance:
                    candidates.append((distance, side))
            if candidates:
                labels[index] = min(candidates)[1]

        assignments: dict[int, tuple[str, float]] = {}
        for side, chain in (("blue", self.left_chain), ("yellow", self.right_chain)):
            sector = self.sector(self.sector_tail_points(chain))
            if sector is None:
                continue
            origin, direction = sector
            for index, point in enumerate(cones):
                if labels[index] != "unknown":
                    continue
                score = self.sector_score(origin, direction, point)
                if score is None:
                    continue
                previous = assignments.get(index)
                if previous is None or score < previous[1]:
                    assignments[index] = (side, score)
        for index, (side, _) in assignments.items():
            labels[index] = side
        return labels

    def can_reinitialize(self, cones: Sequence[Point2D]) -> bool:
        """Require plausible current observations on both sides before a reset."""
        minimum = int(self.parameter("initial_chain_min_length"))
        left_count = sum(self.point_matches_side(point, "blue") for point in cones)
        right_count = sum(self.point_matches_side(point, "yellow") for point in cones)
        return left_count >= minimum and right_count >= minimum

    def closest_seed_candidate(
        self, cones: Sequence[Point2D], side: str
    ) -> int | None:
        minimum_x = float(self.parameter("seed_min_x"))
        candidates = [
            (math.hypot(point[0], point[1]), index)
            for index, point in enumerate(cones)
            if point[0] > minimum_x and self.point_matches_side(point, side)
        ]
        return min(candidates)[1] if candidates else None

    @staticmethod
    def closest_unknown_to_origin(
        cones: Sequence[Point2D],
        labels: Sequence[str],
        origin: Point2D,
        maximum: float,
    ) -> tuple[int, float] | None:
        candidates = [
            (math.dist(origin, point), index)
            for index, point in enumerate(cones)
            if labels[index] == "unknown" and math.dist(origin, point) <= maximum
        ]
        if not candidates:
            return None
        distance, index = min(candidates)
        return index, distance

    @staticmethod
    def sector(points: Sequence[Point2D]):
        if len(points) < 2:
            return None
        ordered = sorted(points, key=lambda point: point[0])
        origin = np.asarray(ordered[-1], dtype=float)
        direction = origin - np.asarray(ordered[-2], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-6:
            return None
        return (float(origin[0]), float(origin[1])), direction / norm

    def sector_score(self, origin: Point2D, direction: np.ndarray, point: Point2D):
        vector = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
        distance = float(np.linalg.norm(vector))
        radius = float(self.parameter("side_sector_radius"))
        if distance <= 1.0e-6 or distance > radius:
            return None
        progress = float(np.dot(direction, vector / distance))
        if progress <= 0.0:
            return None
        angle = math.degrees(math.acos(np.clip(progress, -1.0, 1.0)))
        limit = float(self.parameter("side_sector_angle_deg"))
        if angle > limit:
            return None
        return angle / limit + 0.35 * distance / radius

    def point_matches_side(self, point: Point2D, side: str) -> bool:
        threshold = float(self.parameter("seed_min_abs_y"))
        return point[1] > threshold if side == "blue" else point[1] < -threshold

    def update_chains_from_labels(
        self, cones: Sequence[Point2D], labels: Sequence[str]
    ) -> None:
        for side, chain in (("blue", self.left_chain), ("yellow", self.right_chain)):
            points = [point for point, label in zip(cones, labels) if label == side]
            self.update_side_chain(chain, points)

    def update_side_chain(
        self, chain: list[ChainPoint], points: Sequence[Point2D]
    ) -> None:
        duplicate = float(self.parameter("side_chain_duplicate_distance"))
        for point in sorted(points, key=lambda item: item[0]):
            closest = self.closest_chain_index(chain, point)
            if (
                closest is not None
                and math.dist(self.chain_xy(chain[closest]), point) <= duplicate
            ):
                chain[closest] = (point[0], point[1], 0)
            else:
                chain.append((point[0], point[1], 0))

    def prune_chains(self) -> None:
        behind = float(self.parameter("chain_prune_behind_x"))
        maximum = max(2, int(self.parameter("max_chain_points")))
        self.left_chain = self.pruned_chain(self.left_chain, behind, maximum)
        self.right_chain = self.pruned_chain(self.right_chain, behind, maximum)

    @staticmethod
    def pruned_chain(
        chain: Sequence[ChainPoint], behind: float, maximum: int
    ) -> list[ChainPoint]:
        retained = [point for point in chain if point[0] >= behind]
        retained.sort(key=lambda point: point[0])
        return retained[-maximum:]

    def age_chains(self) -> None:
        self.left_chain = [(x, y, age + 1) for x, y, age in self.left_chain]
        self.right_chain = [(x, y, age + 1) for x, y, age in self.right_chain]

    def compensate_chains_to_time(self, cloud_time: float) -> bool:
        """Move retained chains, or request LiDAR-only rebuilding after a yaw jump."""
        pose = self.interpolate_odometry_pose(cloud_time)
        if pose is None:
            self.last_motion_debug = "motion wait_ekf_pose"
            return self.yaw_realigning
        if self.previous_motion_time is None or self.previous_motion_pose is None:
            self.previous_motion_time = cloud_time
            self.previous_motion_pose = pose
            self.last_motion_debug = "motion init"
            return self.yaw_realigning
        dt = cloud_time - self.previous_motion_time
        if dt < float(self.parameter("motion_min_dt")):
            return self.yaw_realigning
        previous_x, previous_y, previous_yaw = self.previous_motion_pose
        current_x, current_y, current_yaw = pose
        world_dx = current_x - previous_x
        world_dy = current_y - previous_y
        cosine = math.cos(previous_yaw)
        sine = math.sin(previous_yaw)
        dx = cosine * world_dx + sine * world_dy
        dy = -sine * world_dx + cosine * world_dy
        dtheta = self.normalize_angle(current_yaw - previous_yaw)
        dtheta_deg = abs(math.degrees(dtheta))
        jump_threshold = float(self.parameter("yaw_jump_threshold_deg"))

        if dtheta_deg >= jump_threshold:
            # The straight stop mission cannot physically rotate this much in
            # one LiDAR period. Treat it as EKF heading-frame realignment, do
            # not rotate retained cones, and rebuild from the current scan.
            self.left_chain.clear()
            self.right_chain.clear()
            self.virtual_left.clear()
            self.virtual_right.clear()
            self.initialized = False
            self.tracking_loss_count = 0
            self.yaw_realigning = True
            self.yaw_stable_count = 0
            self.previous_motion_time = cloud_time
            self.previous_motion_pose = pose
            self.last_motion_debug = (
                f"motion yaw_jump={math.degrees(dtheta):.2f}deg "
                "lidar_reclassify"
            )
            self.get_logger().warn(
                f"EKF yaw jumped {math.degrees(dtheta):.2f} deg between LiDAR "
                "frames; rebuilding cone sides from current LiDAR",
                throttle_duration_sec=0.5,
            )
            return True

        if self.yaw_realigning:
            stable_threshold = float(self.parameter("yaw_stable_threshold_deg"))
            if dtheta_deg <= stable_threshold:
                self.yaw_stable_count += 1
            else:
                self.yaw_stable_count = 0
            required = max(1, int(self.parameter("yaw_stable_frames")))
            self.previous_motion_time = cloud_time
            self.previous_motion_pose = pose
            if self.yaw_stable_count >= required:
                self.yaw_realigning = False
                self.last_motion_debug = (
                    f"motion yaw_stable={self.yaw_stable_count}/{required} "
                    "lidar_reclassify_final"
                )
            else:
                self.last_motion_debug = (
                    f"motion yaw_realigning dtheta={math.degrees(dtheta):.2f} "
                    f"stable={self.yaw_stable_count}/{required} lidar_reclassify"
                )
            # Reclassify the final stable frame as well. The next cloud starts
            # normal persistent tracking from this current-frame chain.
            return True

        self.left_chain = [
            self.compensate_chain_point(point, dx, dy, dtheta)
            for point in self.left_chain
        ]
        self.right_chain = [
            self.compensate_chain_point(point, dx, dy, dtheta)
            for point in self.right_chain
        ]
        self.previous_motion_time = cloud_time
        self.previous_motion_pose = pose
        self.last_motion_debug = (
            f"motion_ekf_pose dt={dt:.3f} dx={dx:.3f} dy={dy:.3f} "
            f"dtheta={math.degrees(dtheta):.2f}"
        )
        return False

    def interpolate_odometry_pose(self, stamp: float) -> Pose2D | None:
        if not self.odometry_buffer:
            return None
        samples = list(self.odometry_buffer)
        if stamp <= samples[0][0]:
            return samples[0][1], samples[0][2], samples[0][3]
        if stamp >= samples[-1][0]:
            return samples[-1][1], samples[-1][2], samples[-1][3]
        for first, second in zip(samples[:-1], samples[1:]):
            if first[0] <= stamp <= second[0]:
                span = max(second[0] - first[0], 1.0e-6)
                ratio = (stamp - first[0]) / span
                yaw_delta = self.normalize_angle(second[3] - first[3])
                return (
                    first[1] + ratio * (second[1] - first[1]),
                    first[2] + ratio * (second[2] - first[2]),
                    self.normalize_angle(first[3] + ratio * yaw_delta),
                )
        return samples[-1][1], samples[-1][2], samples[-1][3]

    @staticmethod
    def compensate_chain_point(
        point: ChainPoint, dx: float, dy: float, dtheta: float
    ) -> ChainPoint:
        x = point[0] - dx
        y = point[1] - dy
        cosine = math.cos(-dtheta)
        sine = math.sin(-dtheta)
        return (
            cosine * x - sine * y,
            sine * x + cosine * y,
            point[2],
        )

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def closest_chain_index(
        chain: Sequence[ChainPoint], point: Point2D
    ) -> int | None:
        if not chain:
            return None
        return min(
            range(len(chain)),
            key=lambda index: math.dist(
                ConePerceptionNode.chain_xy(chain[index]), point
            ),
        )

    @staticmethod
    def chain_xy(point: ChainPoint) -> Point2D:
        return point[0], point[1]

    @staticmethod
    def points_to_chain(points: Sequence[Point2D]) -> list[ChainPoint]:
        return [(point[0], point[1], 0) for point in points]

    def sector_tail_points(self, chain: Sequence[ChainPoint]) -> list[Point2D]:
        size = max(2, int(self.parameter("sector_tail_size")))
        points = sorted(
            [self.chain_xy(point) for point in chain], key=lambda point: point[0]
        )
        return points[-size:]

    @staticmethod
    def front_chain_points(chain: Sequence[ChainPoint]) -> list[Point2D]:
        return [(x, y) for x, y, _ in chain if x >= 0.0]

    def update_virtual_cones(
        self, measured_left: Sequence[Point2D], measured_right: Sequence[Point2D]
    ) -> None:
        self.virtual_left = []
        self.virtual_right = []
        if not bool(self.parameter("enable_virtual_cones")):
            return
        left = [(x, y) for x, y in measured_left if x >= 0.0]
        right = [(x, y) for x, y in measured_right if x >= 0.0]
        threshold = int(self.parameter("virtual_count_diff_threshold"))
        if len(left) - len(right) >= threshold:
            self.virtual_right = self.virtual_side(left, True, len(left) - len(right))
        elif len(right) - len(left) >= threshold:
            self.virtual_left = self.virtual_side(right, False, len(right) - len(left))

    def virtual_side(
        self, source: Sequence[Point2D], source_is_left: bool, requested: int
    ) -> list[Point2D]:
        if len(source) < 2:
            return []
        count = min(requested, int(self.parameter("virtual_max_count")))
        width = float(self.parameter("virtual_track_width"))
        ordered = sorted(source, key=lambda point: point[0])
        result = []
        for index in range(max(1, len(ordered) - count), len(ordered)):
            vector = np.asarray(ordered[index]) - np.asarray(ordered[index - 1])
            norm = float(np.linalg.norm(vector))
            if norm < 1.0e-6:
                continue
            normal = np.asarray([-vector[1], vector[0]]) / norm
            sign = -1.0 if source_is_left else 1.0
            point = np.asarray(ordered[index]) + sign * width * normal
            result.append((float(point[0]), float(point[1])))
        return result

    @staticmethod
    def closest_index(cones: Sequence[Point2D], predicate):
        candidates = [
            (math.hypot(point[0], point[1]), index)
            for index, point in enumerate(cones)
            if predicate(point)
        ]
        return min(candidates)[1] if candidates else None

    def marker_array(
        self,
        points: Sequence[Point2D],
        labels: Sequence[str],
        frame: str,
        stamp,
        stream: str,
        namespace_prefix: str = "",
    ) -> MarkerArray:
        array = MarkerArray()
        ids = {"blue": 0, "yellow": 0, "raw": 0}
        for point, label in zip(points, labels):
            marker = Marker()
            marker.header.frame_id = frame
            marker.header.stamp = stamp
            marker.ns = namespace_prefix + label
            marker.id = ids[label]
            ids[label] += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = float(self.parameter("cone_height")) * 0.5
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(self.parameter("cone_diameter"))
            marker.scale.y = marker.scale.x
            marker.scale.z = float(self.parameter("cone_height"))
            if stream == "virtual":
                # Inferred, not measured: draw it translucent and slimmer so it
                # never reads as a real detection.
                marker.scale.x *= 0.7
                marker.scale.y = marker.scale.x
                marker.color.a = 0.35
            else:
                marker.color.a = 0.8 if label != "raw" else 0.3
            if label == "blue":
                marker.color.b = 1.0
                marker.color.g = 0.3
            elif label == "yellow":
                marker.color.r = 1.0
                marker.color.g = 0.85
            else:
                marker.color.r = marker.color.g = marker.color.b = 1.0
            array.markers.append(marker)

        # Keep stable namespace/IDs and delete only markers that disappeared from
        # this frame. A DELETEALL followed by ADD on every LiDAR callback can be
        # rendered by RViz as a clear/repaint flash.
        previous = self.published_marker_counts[stream]
        for label, previous_count in previous.items():
            for marker_id in range(ids[label], previous_count):
                marker = Marker()
                marker.header.frame_id = frame
                marker.header.stamp = stamp
                marker.ns = namespace_prefix + label
                marker.id = marker_id
                marker.action = Marker.DELETE
                array.markers.append(marker)
        self.published_marker_counts[stream] = ids.copy()
        return array

    def virtual_marker_array(self, frame: str, stamp) -> MarkerArray:
        """Show the synthetic cones mirrored onto the side that went missing."""
        points = list(self.virtual_left) + list(self.virtual_right)
        labels = ["blue"] * len(self.virtual_left)
        labels.extend(["yellow"] * len(self.virtual_right))
        return self.marker_array(
            points, labels, frame, stamp, "virtual", namespace_prefix="virtual_"
        )

    def sector_marker_array(self, frame: str, stamp) -> MarkerArray:
        """Visualize the final fan-shaped search sector for each current line."""
        array = MarkerArray()

        for side, points in (
            ("blue", self.left_chain),
            ("yellow", self.right_chain),
        ):
            sector = self.sector(self.sector_tail_points(points))
            if sector is None:
                marker = self.new_sector_marker(frame, stamp, side, "outline")
                marker.action = Marker.DELETE
                array.markers.append(marker)
                continue
            origin, direction = sector
            arc = self.sector_arc(origin, direction)
            outline = self.new_sector_marker(frame, stamp, side, "outline")
            outline.type = Marker.LINE_STRIP
            outline.scale.x = 0.06
            outline.color.a = 0.9
            outline.points = [self.marker_point(origin)]
            outline.points.extend(self.marker_point(point) for point in arc)
            outline.points.append(self.marker_point(origin))
            array.markers.append(outline)
        return array

    def sector_arc(
        self, origin: Point2D, direction: np.ndarray
    ) -> list[Point2D]:
        radius = float(self.parameter("side_sector_radius"))
        limit = math.radians(float(self.parameter("side_sector_angle_deg")))
        arc = []
        for angle in np.linspace(-limit, limit, 25):
            cosine = math.cos(float(angle))
            sine = math.sin(float(angle))
            rotated = np.asarray([
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
            ])
            point = np.asarray(origin) + radius * rotated
            arc.append((float(point[0]), float(point[1])))
        return arc

    @staticmethod
    def marker_point(point: Point2D) -> Point:
        result = Point()
        result.x = point[0]
        result.y = point[1]
        result.z = 0.08
        return result

    @staticmethod
    def new_sector_marker(frame: str, stamp, side: str, kind: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = f"{side}_search_{kind}"
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        if side == "blue":
            marker.color.b = 1.0
            marker.color.g = 0.3
        else:
            marker.color.r = 1.0
            marker.color.g = 0.85
        return marker

    def publish_status(
        self, raw_count: int, roi_count: int, cone_count: int, labels: Sequence[str]
    ) -> None:
        message = String()
        message.data = (
            f"raw_points={raw_count} roi_points={roi_count} cones={cone_count} "
            f"left={labels.count('blue')} right={labels.count('yellow')} "
            f"left_chain={len(self.left_chain)} right_chain={len(self.right_chain)} "
            f"initialized={self.initialized} tracking_loss={self.tracking_loss_count} "
            f"yaw_realigning={self.yaw_realigning} {self.last_motion_debug}"
        )
        self.status_pub.publish(message)

    def message_time(self, stamp) -> float:
        value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        return value if value > 0.0 else self.get_clock().now().nanoseconds * 1.0e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConePerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

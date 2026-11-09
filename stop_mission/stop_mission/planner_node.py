"""Build a local center path from the tracked left/right cone markers."""

from __future__ import annotations

import math
from typing import Sequence

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from .algorithms import (
    Point2D,
    delaunay_midpoint_graph,
    nearest,
    path_yaw_curvature,
    resample_path,
)


class ConePlannerNode(Node):
    """Delaunay midpoint planner copied from xycar_ws and adapted to KROWD topics."""

    def __init__(self) -> None:
        super().__init__("stop_mission_cone_planner")
        defaults = {
            "marker_topic": "/stop_mission/cones",
            "path_topic": "/stop_mission/path",
            "delaunay_topic": "/stop_mission/delaunay",
            "end_distance_topic": "/stop_mission/end_distance",
            "status_topic": "/stop_mission/planner_status",
            "start_topic": "/stop_mission/start",
            "auto_start": True,
            "frame_id": "base_link",
            "publish_rate": 20.0,
            "min_inner_angle_deg": 10.0,
            "max_inner_angle_deg": 160.0,
            "max_turn_angle_deg": 120.0,
            "one_side_offset": 3.0,
            "path_spacing": 0.20,
            "extension_length": 4.0,
            "marker_timeout": 0.5,
            "time_rewind_tolerance_sec": 0.1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.started = bool(self.parameter("auto_start"))
        self.points: list[Point2D] = []
        self.colors: list[str] = []
        self.input_stamp = None
        self.input_frame = str(self.parameter("frame_id"))
        self.last_marker_time = None
        self.last_input_stamp_sec: float | None = None

        self.create_subscription(
            MarkerArray,
            str(self.parameter("marker_topic")),
            self.marker_callback,
            10,
        )
        self.create_subscription(
            Bool, str(self.parameter("start_topic")), self.start_callback, 10
        )
        self.path_pub = self.create_publisher(
            Path, str(self.parameter("path_topic")), 10
        )
        self.delaunay_pub = self.create_publisher(
            Marker, str(self.parameter("delaunay_topic")), 10
        )
        self.end_distance_pub = self.create_publisher(
            Float32, str(self.parameter("end_distance_topic")), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.parameter("status_topic")), 10
        )
        self.create_timer(
            1.0 / max(float(self.parameter("publish_rate")), 1.0), self.plan_once
        )

    def parameter(self, name: str):
        return self.get_parameter(name).value

    def start_callback(self, message: Bool) -> None:
        self.started = bool(message.data)
        if not self.started:
            self.reset_input()

    def reset_input(self) -> None:
        self.points.clear()
        self.colors.clear()
        self.input_stamp = None
        self.last_marker_time = None
        self.last_input_stamp_sec = None

    def marker_callback(self, message: MarkerArray) -> None:
        if not self.started:
            return
        message_stamp = next(
            (marker.header.stamp for marker in message.markers), None
        )
        if message_stamp is not None:
            stamp_sec = float(message_stamp.sec) + float(message_stamp.nanosec) * 1.0e-9
            tolerance = float(self.parameter("time_rewind_tolerance_sec"))
            if (
                self.last_input_stamp_sec is not None
                and stamp_sec < self.last_input_stamp_sec - tolerance
            ):
                self.get_logger().info(
                    "Cone-marker time moved backwards; resetting planner input"
                )
                self.reset_input()
            self.last_input_stamp_sec = stamp_sec
        points: list[Point2D] = []
        colors: list[str] = []
        for marker in message.markers:
            if marker.action != Marker.ADD or marker.ns not in ("blue", "yellow"):
                continue
            points.append((float(marker.pose.position.x), float(marker.pose.position.y)))
            colors.append(marker.ns)
            self.input_stamp = marker.header.stamp
            self.input_frame = marker.header.frame_id or str(self.parameter("frame_id"))
        self.points = points
        self.colors = colors
        self.last_marker_time = self.get_clock().now()

    def plan_once(self) -> None:
        if not self.started or not self.points or self.markers_are_stale():
            return
        forward = [
            (point, color)
            for point, color in zip(self.points, self.colors)
            if point[0] > 0.0
        ]
        if not forward:
            return
        points = np.asarray([item[0] for item in forward], dtype=float)
        colors = [item[1] for item in forward]
        end_distance = max(float(point[0]) for point in points)
        path, accepted = self.build_path(points, colors)
        if len(path) < 2:
            return
        path = self.extend_path(path)
        path = resample_path(path, float(self.parameter("path_spacing")))
        self.path_pub.publish(self.path_message(path))
        self.end_distance_pub.publish(Float32(data=end_distance))
        self.delaunay_pub.publish(self.delaunay_marker(points, accepted))
        status = String()
        status.data = (
            f"cones={len(points)} triangles={len(accepted)} path_points={len(path)}"
        )
        self.status_pub.publish(status)

    def markers_are_stale(self) -> bool:
        if self.last_marker_time is None:
            return True
        age = (self.get_clock().now() - self.last_marker_time).nanoseconds * 1.0e-9
        return age < 0.0 or age > float(self.parameter("marker_timeout"))

    def build_path(
        self, points: np.ndarray, colors: Sequence[str]
    ) -> tuple[list[Point2D], list[np.ndarray]]:
        color_set = set(colors)
        if len(color_set) == 1:
            return self.one_side_path(points, next(iter(color_set))), []
        if len(points) < 3:
            return self.paired_midline(points, colors), []

        graph, midpoint_info, accepted = delaunay_midpoint_graph(
            points,
            colors,
            float(self.parameter("min_inner_angle_deg")),
            float(self.parameter("max_inner_angle_deg")),
        )
        start = nearest(graph, (0.0, 0.0))
        if start is None:
            return self.paired_midline(points, colors), accepted

        graph.setdefault((0.0, 0.0), {})[start] = math.dist((0.0, 0.0), start)
        graph.setdefault(start, {})[(0.0, 0.0)] = math.dist((0.0, 0.0), start)
        path = [(0.0, 0.0), start]
        visited = set(path)
        while True:
            current = path[-1]
            candidates = []
            for candidate in graph.get(current, {}):
                if candidate in visited:
                    continue
                turn = self.turn_angle(path[-2], current, candidate)
                if turn > float(self.parameter("max_turn_angle_deg")):
                    continue
                color_penalty = self.color_penalty(current, candidate, midpoint_info)
                backward_penalty = 1.0 if candidate[0] < current[0] - 0.25 else 0.0
                score = turn / 180.0 + 1.3 * color_penalty + backward_penalty
                candidates.append((score, candidate))
            if not candidates:
                break
            _, selected = min(candidates, key=lambda item: item[0])
            path.append(selected)
            visited.add(selected)

        if len(path) < 3:
            fallback = self.paired_midline(points, colors)
            return fallback if len(fallback) >= len(path) else path, accepted
        return path, accepted

    def paired_midline(
        self, points: np.ndarray, colors: Sequence[str]
    ) -> list[Point2D]:
        blue = sorted(
            [point for point, color in zip(points, colors) if color == "blue"],
            key=lambda point: (point[0], abs(point[1])),
        )
        yellow = sorted(
            [point for point, color in zip(points, colors) if color == "yellow"],
            key=lambda point: (point[0], abs(point[1])),
        )
        if blue and yellow:
            result: list[Point2D] = [(0.0, 0.0)]
            for left, right in zip(blue, yellow):
                midpoint = (np.asarray(left) + np.asarray(right)) * 0.5
                result.append((float(midpoint[0]), float(midpoint[1])))
            return self.remove_duplicates(result)
        if blue:
            return self.one_side_path(np.asarray(blue), "blue")
        if yellow:
            return self.one_side_path(np.asarray(yellow), "yellow")
        return []

    def one_side_path(self, points: np.ndarray, color: str) -> list[Point2D]:
        ordered = np.asarray(sorted(points, key=lambda point: point[0]), dtype=float)
        if len(ordered) < 2:
            return []
        side = -1.0 if color == "blue" else 1.0
        offset = float(self.parameter("one_side_offset"))
        result: list[Point2D] = [(0.0, 0.0)]
        for index, point in enumerate(ordered):
            vector = ordered[min(index + 1, len(ordered) - 1)] - ordered[max(index - 1, 0)]
            norm = float(np.linalg.norm(vector))
            if norm < 1.0e-6:
                continue
            normal = np.asarray([-vector[1], vector[0]]) / norm
            center = point + side * offset * normal
            result.append((float(center[0]), float(center[1])))
        return self.remove_duplicates(result)

    @staticmethod
    def color_penalty(current, candidate, midpoint_info) -> float:
        if candidate not in midpoint_info:
            return 0.0
        first, second, first_color, second_color = midpoint_info[candidate]
        direction = np.asarray(candidate) - np.asarray(current)
        penalty = 0.0
        for point, color in ((first, first_color), (second, second_color)):
            cross = float(np.cross(direction, point - np.asarray(candidate)))
            if (cross > 0.0 and color == "yellow") or (
                cross < 0.0 and color == "blue"
            ):
                penalty += 0.5
        return penalty

    @staticmethod
    def turn_angle(first: Point2D, second: Point2D, third: Point2D) -> float:
        vector1 = np.asarray(second) - np.asarray(first)
        vector2 = np.asarray(third) - np.asarray(second)
        norms = float(np.linalg.norm(vector1) * np.linalg.norm(vector2))
        if norms < 1.0e-9:
            return 0.0
        cosine = np.clip(float(np.dot(vector1, vector2)) / norms, -1.0, 1.0)
        return float(math.degrees(math.acos(cosine)))

    def extend_path(self, path: Sequence[Point2D]) -> list[Point2D]:
        result = list(path)
        if len(result) < 2:
            return result
        vector = np.asarray(result[-1]) - np.asarray(result[-2])
        norm = float(np.linalg.norm(vector))
        if norm > 1.0e-6:
            extended = np.asarray(result[-1]) + vector / norm * float(
                self.parameter("extension_length")
            )
            result.append((float(extended[0]), float(extended[1])))
        return result

    @staticmethod
    def remove_duplicates(points: Sequence[Point2D]) -> list[Point2D]:
        result = []
        for point in points:
            if not result or math.dist(result[-1], point) >= 0.05:
                result.append(point)
        return result

    def path_message(self, path: Sequence[Point2D]) -> Path:
        stamp = self.input_stamp or self.get_clock().now().to_msg()
        frame = self.input_frame or str(self.parameter("frame_id"))
        yaw, _ = path_yaw_curvature(path)
        message = Path()
        message.header.frame_id = frame
        message.header.stamp = stamp
        for (x, y), heading in zip(path, yaw):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(float(heading) * 0.5)
            pose.pose.orientation.w = math.cos(float(heading) * 0.5)
            message.poses.append(pose)
        return message

    def delaunay_marker(
        self, points: np.ndarray, accepted: Sequence[np.ndarray]
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.input_frame
        marker.header.stamp = self.input_stamp or self.get_clock().now().to_msg()
        marker.ns = "delaunay"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.r = marker.color.g = marker.color.b = 1.0
        marker.color.a = 0.5
        marker.pose.orientation.w = 1.0
        for simplex in accepted:
            for i in range(3):
                for j in range(i + 1, 3):
                    for index in (int(simplex[i]), int(simplex[j])):
                        point = Point()
                        point.x = float(points[index][0])
                        point.y = float(points[index][1])
                        marker.points.append(point)
        return marker


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConePlannerNode()
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

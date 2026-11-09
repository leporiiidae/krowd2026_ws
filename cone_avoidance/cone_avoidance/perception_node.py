"""Cone perception node with LiDAR detection and tracking."""

from __future__ import annotations

"""LiDAR cone detection on a base_link point cloud (numpy / scipy only)."""


import math
from typing import List

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def min_points(dist: float, min_pts_at_5m: float) -> int:
    """Point-count threshold falls with range squared (16-ch ring spacing)."""
    return max(2, int(round(min_pts_at_5m * (5.0 / max(dist, 0.5)) ** 2)))


def deskew(xyz: np.ndarray, t_rel: np.ndarray, v: float, yaw_rate: float) -> np.ndarray:
    """Express every point in the base_link pose at t_rel = 0.

    A point measured ``t`` seconds later was seen from a pose advanced by the
    vehicle motion, so it is mapped back through that motion (planar, constant
    twist over one scan).
    """
    theta = yaw_rate * t_rel
    c, s = np.cos(theta), np.sin(theta)
    # Pose advance over t (arc; reduces to v*t straight ahead when yaw_rate -> 0).
    px = v * t_rel * np.sinc(theta / np.pi)
    py = v * t_rel * 0.5 * theta * np.sinc(theta / (2 * np.pi)) ** 2
    out = xyz.copy()
    out[:, 0] = c * xyz[:, 0] - s * xyz[:, 1] + px
    out[:, 1] = s * xyz[:, 0] + c * xyz[:, 1] + py
    return out


def cluster_xy(xy: np.ndarray, eps: float) -> np.ndarray:
    """Single-link clustering: labels of connected components under distance eps."""
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=int)
    pairs = cKDTree(xy).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def fit_center(xy: np.ndarray, radius: float) -> np.ndarray:
    """Centre of a known-radius cylinder from points on its sensor-facing arc.

    The centroid of the visible arc sits ~0.785 r in front of the axis, so the
    start value is pushed along the ray; a few Gauss-Newton steps refine it.
    """
    centroid = xy.mean(axis=0)
    ray = centroid / max(np.linalg.norm(centroid), 1e-6)
    center = centroid + 0.785 * radius * ray
    if len(xy) < 3:
        return center
    for _ in range(4):
        diff = xy - center
        rng = np.maximum(np.linalg.norm(diff, axis=1), 1e-6)
        jac = -diff / rng[:, None]
        res = rng - radius
        step, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        if not np.all(np.isfinite(step)) or np.linalg.norm(step) > radius:
            break  # ill-conditioned (nearly collinear points): keep the ray-biased start
        center = center + step
    return center


def _width(xy: np.ndarray) -> float:
    """Extent perpendicular to the sensor ray."""
    centroid = xy.mean(axis=0)
    ray = centroid / max(np.linalg.norm(centroid), 1e-6)
    lateral = xy @ np.array([-ray[1], ray[0]])
    return float(lateral.max() - lateral.min())


def _row_axis(xy: np.ndarray) -> np.ndarray:
    """Unit vector along a row of cones (principal axis of the cluster)."""
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0]


def _split_row(xy: np.ndarray, cfg) -> List[np.ndarray]:
    """Cone centres of a cluster wider than one cone.

    Cones standing side by side (touching, or closer than cluster_eps) arrive as ONE
    cluster. The count comes from the extent along the row, the members from 1-D
    k-means along it. Afterwards the two end cones are pushed outward, if needed,
    until they cover the cluster's extreme points: the planner steers by the inner
    edge of the row, so the row's extent must never be under-estimated.
    """
    p = cfg.perception
    radius = p.cone_diameter / 2.0
    axis = _row_axis(xy)
    t = xy @ axis
    extent = float(t.max() - t.min())
    # Real rows are not perfectly tight (measured 1.52 m for three 0.45 m cones).
    k = max(2, int(round(extent / (p.cone_diameter * p.row_pitch_ratio))))
    if k > int(p.max_cones_per_cluster):
        return []  # wall / guardrail, not a row of cones
    centers = np.linspace(t.min(), t.max(), 2 * k + 1)[1::2]
    for _ in range(8):
        label = np.argmin(np.abs(t[:, None] - centers[None, :]), axis=1)
        centers = np.array([t[label == j].mean() if np.any(label == j) else centers[j]
                            for j in range(k)])
    order = np.argsort(centers)
    fitted = []
    for j in order:
        part = xy[label == j]
        if len(part) == 0:
            continue
        dist = float(np.linalg.norm(part.mean(axis=0)))
        if len(part) < min_points(dist, p.min_pts_at_5m):
            continue
        fitted.append(fit_center(part, radius))
    if not fitted:
        return []
    low = float(fitted[0] @ axis) - radius - float(t.min())
    if low > 0.0:
        fitted[0] = fitted[0] - low * axis
    high = float(t.max()) - (float(fitted[-1] @ axis) + radius)
    if high > 0.0:
        fitted[-1] = fitted[-1] + high * axis
    return fitted


def _looks_like_cone(center: np.ndarray, column: np.ndarray, cfg) -> bool:
    """Height test on ALL returns above the footprint (z relative to the ground).

    Roadside vegetation and posts pass the width test all the time; what separates
    them is height: a cone ends at cone_height, and (close enough for the rings to be
    dense) it does reach up to near cone_height.
    """
    p = cfg.perception
    if len(column) == 0:
        return True
    top = float(column.max())
    if top > p.cone_height + p.tall_reject_margin:
        return False                                   # something taller stands here
    dist = float(np.linalg.norm(center))
    if dist < p.top_check_range and top < p.cone_height - p.top_min_below:
        return False                                   # too short to be a cone
    return True


def detect_cones(xyz: np.ndarray, cfg, ground_z: float = 0.0) -> np.ndarray:
    """Cone axis positions (M, 2) in base_link from one (deskewed) scan."""
    p = cfg.perception
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2] - ground_z
    sx0, sx1, sy = p.self_box
    roi = ((x >= p.roi_x[0]) & (x <= p.roi_x[1]) & (y >= p.roi_y[0]) & (y <= p.roi_y[1])
           & np.isfinite(x) & ~((x >= sx0) & (x <= sx1) & (np.abs(y) <= sy)))   # own bodywork
    keep = roi & (z >= p.ground_z[0]) & (z <= p.ground_z[1])
    xy = xyz[keep, :2]
    if len(xy) == 0:
        return np.zeros((0, 2))
    # Everything above the ground band, for the height test (includes tall objects).
    above = roi & (z >= p.ground_z[0]) & (z <= p.cone_height + 2.0)
    col_xy, col_z = xyz[above, :2], z[above]
    col_tree = cKDTree(col_xy) if len(col_xy) else None

    diameter = p.cone_diameter
    w_min, w_max = p.width_ratio_min * diameter, p.width_ratio_max * diameter
    labels = cluster_xy(xy, p.cluster_eps)
    out = []
    for label in np.unique(labels):
        pts = xy[labels == label]
        axis = _row_axis(pts) if len(pts) >= 3 else np.array([0.0, 1.0])
        along = pts @ axis
        if max(_width(pts), float(along.max() - along.min())) > w_max:
            candidates = _split_row(pts, cfg)
        else:
            dist = float(np.linalg.norm(pts.mean(axis=0)))
            if len(pts) < min_points(dist, p.min_pts_at_5m):
                continue
            # Far cones are hit by very few columns, so only the upper bound is firm.
            if _width(pts) < w_min and dist < 8.0:
                continue
            candidates = [fit_center(pts, diameter / 2.0)]
        for center in candidates:
            idx = col_tree.query_ball_point(center, diameter * 0.75) if col_tree is not None else []
            if _looks_like_cone(center, col_z[idx], cfg):
                out.append(center)
    return np.array(out).reshape(-1, 2)


def synth_scan(cones_xy, radius: float, sensor_z: float = 0.8, height: float = 0.8,
               az_step_deg: float = 0.35, rings_deg=None, max_range: float = 25.0) -> np.ndarray:
    """Ray-cast cylinders from a sensor at the origin (testing / sim_vehicle only).

    Every ray keeps its NEAREST hit, so cones standing shoulder to shoulder occlude
    each other the way they do for a real spinning LiDAR.
    """
    cones = np.asarray(cones_xy, dtype=float).reshape(-1, 2)
    rings = np.radians(np.linspace(-15.0, 15.0, 16) if rings_deg is None else rings_deg)
    if len(cones) == 0:
        return np.zeros((0, 3))
    az = np.arange(-math.pi / 2, math.pi / 2, math.radians(az_step_deg))
    ux, uy = np.cos(az), np.sin(az)
    b = ux[:, None] * cones[None, :, 0] + uy[:, None] * cones[None, :, 1]
    c = (cones ** 2).sum(axis=1)[None, :] - radius * radius
    disc = b * b - c
    rng = np.where((disc >= 0) & (b > 0), b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
    rng = np.where(rng > 0, rng, np.inf).min(axis=1)
    hit = np.isfinite(rng) & (rng <= max_range)
    pts = []
    for r, x, y in zip(rng[hit], ux[hit], uy[hit]):
        z = sensor_z + r * np.tan(rings)
        for zz in z[(z >= 0.0) & (z <= height)]:
            pts.append((r * x, r * y, float(zz)))
    return np.array(pts).reshape(-1, 3)
"""Cone tracking in the global frame and longitudinal pair grouping."""


from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    hits: int = 1              # consecutive observed frames until confirmed, then total
    misses: int = 0            # consecutive unobserved frames
    confirmed: bool = False
    observations: int = 1

    @property
    def xy(self):
        return (self.x, self.y)


class ConeTracker:
    """Nearest-neighbour tracker with first-order position filtering.

    Cones are static and localisation is RTK-grade, so a CONFIRMED track is never
    dropped: a cone that disappears ahead is a momentary miss (keep trusting the
    estimate), and one that disappears behind is simply out of view. Only
    unconfirmed candidates expire after ``miss_tolerance`` frames.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0

    def update(self, observations_xy: Sequence[Sequence[float]]) -> List[Track]:
        p = self.cfg.perception
        obs = np.asarray(observations_xy, dtype=float).reshape(-1, 2)
        ids = list(self.tracks.keys())
        unmatched_obs = set(range(len(obs)))
        matched_tracks = set()

        if ids and len(obs):
            pos = np.array([self.tracks[i].xy for i in ids])
            dist = np.hypot(pos[:, None, 0] - obs[None, :, 0], pos[:, None, 1] - obs[None, :, 1])
            # Greedy global-nearest assignment: closest pairs first.
            for flat in np.argsort(dist, axis=None):
                ti, oi = np.unravel_index(flat, dist.shape)
                if dist[ti, oi] > p.track_match_distance:
                    break
                if ids[ti] in matched_tracks or oi not in unmatched_obs:
                    continue
                track = self.tracks[ids[ti]]
                a = p.track_filter_alpha
                track.x = (1.0 - a) * track.x + a * obs[oi, 0]
                track.y = (1.0 - a) * track.y + a * obs[oi, 1]
                track.hits += 1
                track.observations += 1
                track.misses = 0
                if track.hits >= p.confirm_frames:
                    track.confirmed = True
                matched_tracks.add(ids[ti])
                unmatched_obs.discard(oi)

        for track_id in ids:
            if track_id in matched_tracks:
                continue
            track = self.tracks[track_id]
            track.misses += 1
            if not track.confirmed:
                track.hits = 0  # promotion requires CONSECUTIVE observations
                if track.misses > p.miss_tolerance:
                    del self.tracks[track_id]

        for oi in sorted(unmatched_obs):
            self.tracks[self._next_id] = Track(self._next_id, float(obs[oi, 0]), float(obs[oi, 1]))
            self._next_id += 1

        return self.confirmed()

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.confirmed]


@dataclass
class ConePair:
    """Cones sharing one longitudinal station (normally two, side by side)."""

    track_ids: List[int]
    s: np.ndarray
    d: np.ndarray

    @property
    def s_min(self) -> float:
        return float(self.s.min())

    @property
    def s_max(self) -> float:
        return float(self.s.max())

    @property
    def s_center(self) -> float:
        return float(self.s.mean())

    @property
    def sd(self) -> np.ndarray:
        return np.column_stack((self.s, self.d))


def group_pairs(track_ids: Sequence[int], cones_sd: np.ndarray, window: float) -> List[ConePair]:
    """Group cones whose s lies within ``window`` of the group's first cone."""
    cones_sd = np.asarray(cones_sd, dtype=float).reshape(-1, 2)
    order = np.argsort(cones_sd[:, 0]) if len(cones_sd) else []
    groups: List[List[int]] = []
    for idx in order:
        if groups and cones_sd[idx, 0] - cones_sd[groups[-1][0], 0] <= window:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [
        ConePair([int(track_ids[i]) for i in g], cones_sd[g, 0].copy(), cones_sd[g, 1].copy())
        for g in groups
    ]


def find_rows(xy: np.ndarray, heading: float, cfg, origin=None) -> List[List[int]]:
    """Indices of cones that form obstacle rows: N cones side by side ACROSS the road.

    ``heading``/``origin``: travel direction (rad) and vehicle position, same frame as
    ``xy``. A row is a chain of cones whose neighbours are at most ``row_gap_max``
    apart, lying across the travel direction, ``row_min_cones``..``row_max_cones`` long.
    When more are chained (clutter fused to a row), the ``cones_per_group`` cones
    nearest the vehicle's line are the row: groups stand against a lane line, so
    anything fused on is on the OUTER side, beyond that line.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    m = cfg.mission
    if len(xy) < m.row_min_cones:
        return []
    origin = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    along_u = np.array([np.cos(heading), np.sin(heading)])
    across_u = np.array([-along_u[1], along_u[0]])
    s = (xy - origin) @ along_u
    d = (xy - origin) @ across_u
    max_len = m.cones_per_group * cfg.perception.cone_diameter + m.row_len_slack

    n = len(xy)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if abs(s[i] - s[j]) <= m.row_along_tol and abs(d[i] - d[j]) <= m.row_gap_max:
                parent[root(i)] = root(j)
    comps = {}
    for i in range(n):
        comps.setdefault(root(i), []).append(i)

    rows = []
    for members in comps.values():
        if len(members) < m.row_min_cones:
            continue
        # Order from the vehicle's line outward, so trimming drops the OUTER cones.
        side = 1.0 if np.mean(d[members]) >= 0 else -1.0
        members = sorted(members, key=lambda i: side * d[i])
        # Touching cones sit one pitch apart; fused clutter breaks that rhythm. Pick the
        # most regular window, so clutter on EITHER end is left out of the row itself.
        pitch = cfg.perception.cone_diameter * cfg.perception.row_pitch_ratio
        best, best_key = None, None
        for size in _sizes(len(members), m):
            for k in range(len(members) - size + 1):
                win = members[k:k + size]
                span = abs(d[win[-1]] - d[win[0]])
                spread = s[win].max() - s[win].min()
                if not (span <= max_len and spread <= m.row_along_tol and _crosses(xy[win], heading, m)):
                    continue
                gaps = np.abs(np.diff(d[win]))
                key = float(np.abs(gaps - pitch).sum())
                if best_key is None or key < best_key:
                    best, best_key = win, key
            if best is not None:
                break
        if best is None:
            continue
        # Fused members INWARD of the row stand in the driving area: they are obstacles
        # too, so they stay. Only those beyond the outer end (shoulder / other lane) go.
        inner_edge = min(side * d[i] for i in best)
        if not _within_reach(inner_edge, float(np.mean(s[best])), cfg):
            continue
        extras = [i for i in members if i not in best and side * d[i] < inner_edge]
        if _isolated(xy, best + extras, s, d, side, m):
            rows.append(best + extras)
    return rows


def _within_reach(inner_edge: float, ahead: float, cfg) -> bool:
    """A group stands inside the lane the vehicle is in; roadside rows are metres off.

    The vehicle's own heading is the only lane reference perception has, and a few
    degrees of yaw shift a far row sideways, so the allowance grows with distance.
    """
    m = cfg.mission
    allowed = m.row_lateral_max + abs(ahead) * np.tan(np.radians(m.row_heading_tol))
    return inner_edge - cfg.cone_radius <= allowed


def _sizes(available: int, m):
    """Window sizes to try: the rule's group size first, then the others."""
    lo, hi = m.row_min_cones, min(available, m.row_max_cones)
    preferred = [m.cones_per_group] if lo <= m.cones_per_group <= hi else []
    return preferred + [k for k in range(hi, lo - 1, -1) if k not in preferred]


def _crosses(xy: np.ndarray, heading: float, m) -> bool:
    """The row lies across the road: its direction is near-perpendicular to travel."""
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    angle = abs(np.arctan2(vt[0, 1], vt[0, 0]) - heading) % np.pi
    return np.degrees(abs(angle - np.pi / 2.0)) <= m.row_angle_tol


def _isolated(xy, members, s, d, side, m) -> bool:
    """Obstacle rows stand on open road; vegetation is a DENSE patch of clutter.

    Counts neighbours on the road side only (in front, behind, inward). Beyond the row's
    OUTER end lies the lane line and then the shoulder or the other lane, where growth
    next to a real group is normal. One or two objects near a row never reject it: an
    object on the road is an obstacle itself, and deleting the row for it would hide
    both. Measured on 0919_avoidance_f: real rows 0 neighbours, clutter rows 4-5.
    A track within ``row_dup_radius`` of a member is the same cone (duplicate track).
    """
    others = np.delete(np.arange(len(xy)), members)
    if len(others) == 0:
        return True
    dist = np.hypot(xy[others, None, 0] - xy[members][None, :, 0],
                    xy[others, None, 1] - xy[members][None, :, 1]).min(axis=1)
    outer_edge = max(side * d[i] for i in members)
    beyond = side * d[others] > outer_edge + 0.5 * m.row_gap_max
    near = (dist > m.row_dup_radius) & (dist < m.row_clear_radius) & ~beyond
    # A partial row (fewer cones than a group) is easier to form by chance in a patch of
    # clutter, so it tolerates fewer neighbours.
    limit = m.row_clutter_min if len(members) >= m.cones_per_group else m.row_clutter_min_partial
    return int(np.sum(near)) < limit


class RowFilter:
    """Keeps only tracks that belong to obstacle rows, and remembers accepted rows.

    Cones do not move, so a row accepted once stays accepted: clutter that shows up next
    to it later (seen from a new angle) cannot make a real obstacle disappear, and the
    row survives while the car turns across it mid-maneuver.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.accepted: set = set()

    def reset(self) -> None:
        self.accepted.clear()

    def update(self, tracks: Sequence[Track], heading: float, origin=None) -> List[Track]:
        if not tracks:
            return []
        xy = np.array([t.xy for t in tracks])
        for row in find_rows(xy, heading, self.cfg, origin):
            self.accepted.update(tracks[i].track_id for i in row)
        return [t for t in tracks if t.track_id in self.accepted]
"""LiDAR cone detection + global-frame tracking."""


import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray, Int32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .csv_logger import CsvLogger
from .ros_common import run_node, OdomState, declare_config, stamp_to_sec, yaw_from_quaternion


def _rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("cone_perception")
        self.cfg = declare_config(self)
        t = self.cfg.topics
        self.tracker = ConeTracker(self.cfg)
        self.rows = RowFilter(self.cfg)
        self.obstacles = []
        self.odom = OdomState(self.cfg)
        self.odom_buffer = deque(maxlen=200)
        self.extrinsic = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.log = CsvLogger(self.cfg.logging.path, "cones", self.cfg.logging.enabled)

        self.create_subscription(Odometry, t.odometry, self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, t.pointcloud, self.on_cloud, qos_profile_sensor_data)
        self.cones_pub = self.create_publisher(Float64MultiArray, t.cones, 10)
        self.raw_pub = self.create_publisher(Float64MultiArray, t.raw_cones, 10)
        self.count_pub = self.create_publisher(Int32, t.lidar_raw_count, 10)
        self.marker_pub = self.create_publisher(MarkerArray, t.cone_markers, 10)
        self.odom_frame = "odom"

    def on_odom(self, msg: Odometry) -> None:
        self.odom.update(msg)
        self.odom_frame = msg.header.frame_id or self.odom_frame
        pose = msg.pose.pose
        self.odom_buffer.append((stamp_to_sec(msg.header.stamp), pose.position.x, pose.position.y,
                                 yaw_from_quaternion(pose.orientation)))

    def _pose_at(self, t: float):
        """Base pose at the scan stamp: interpolate the buffer, else dead-reckon."""
        buf = self.odom_buffer
        for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(buf, list(buf)[1:]):
            if t0 <= t <= t1 and t1 > t0:
                r = (t - t0) / (t1 - t0)
                da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
                return x0 + r * (x1 - x0), y0 + r * (y1 - y0), a0 + r * da
        return self.odom.base_pose_at(t)

    def _extrinsic(self, frame: str):
        f = self.cfg.frames
        if not f.use_tf or frame == f.base_frame:
            x, y, z, roll, pitch, yaw = f.lidar_xyz_rpy
            return _rpy_matrix(roll, pitch, yaw), np.array([x, y, z])
        if self.extrinsic is None:
            try:
                tf = self.tf_buffer.lookup_transform(f.base_frame, frame, rclpy.time.Time())
            except Exception as error:
                self.get_logger().warn(f"waiting for TF {frame}->{f.base_frame}: {error}",
                                       throttle_duration_sec=2.0)
                return None
            tr = tf.transform.translation
            self.extrinsic = (_quat_matrix(tf.transform.rotation), np.array([tr.x, tr.y, tr.z]))
        return self.extrinsic

    def on_cloud(self, msg: PointCloud2) -> None:
        if not self.odom.valid:
            return
        ext = self._extrinsic(msg.header.frame_id)
        if ext is None:
            return
        p = self.cfg.perception
        names = [f.name for f in msg.fields]
        use_time = p.deskew and p.point_time_field in names
        fields = ["x", "y", "z"] + ([p.point_time_field] if use_time else [])
        cloud = point_cloud2.read_points(msg, field_names=fields, skip_nans=True)
        xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(float)
        if len(xyz) == 0:
            return
        rot, trans = ext
        xyz = xyz @ rot.T + trans
        if use_time:
            t_rel = cloud[p.point_time_field].astype(float) * p.point_time_scale
            xyz = deskew(xyz, t_rel, self.odom.v, self.odom.yaw_rate)

        local = detect_cones(xyz, self.cfg, self.cfg.frames.ground_z)
        bx, by, byaw = self._pose_at(stamp_to_sec(msg.header.stamp))
        c, s = math.cos(byaw), math.sin(byaw)
        glob = np.column_stack((bx + c * local[:, 0] - s * local[:, 1],
                                by + s * local[:, 0] + c * local[:, 1])) if len(local) else np.zeros((0, 2))
        in_range = np.linalg.norm(local, axis=1) <= p.detect_range if len(local) else np.zeros(0, bool)
        self.tracker.update(glob[in_range])

        self.get_logger().info(
            f"cloud {len(xyz)} pts -> band {int(np.sum((xyz[:, 2] - self.cfg.frames.ground_z >= p.ground_z[0]) & (xyz[:, 2] - self.cfg.frames.ground_z <= p.ground_z[1])))}"
            f" -> cones {len(local)} (in range {int(in_range.sum())})"
            f" -> tracks {len(self.tracker.tracks)} confirmed {len(self.tracker.confirmed())}"
            f" -> obstacles {len(self.obstacles)}", throttle_duration_sec=2.0)
        self.raw_pub.publish(Float64MultiArray(data=glob.reshape(-1).tolist()))
        self.count_pub.publish(Int32(data=int(len(glob))))
        # Only tracks forming an obstacle row (N cones side by side across the road) go on.
        self.obstacles = self.rows.update(self.tracker.confirmed(), byaw, (bx, by))
        rows = []
        for tr in self.obstacles:
            rows.extend([float(tr.track_id), tr.x, tr.y, float(tr.misses)])
        self.cones_pub.publish(Float64MultiArray(data=rows))
        self._markers(msg.header.stamp)
        self.log.write({"t": stamp_to_sec(msg.header.stamp), "base_x": bx, "base_y": by, "base_yaw": byaw,
                        "raw": ";".join(f"{x:.3f},{y:.3f}" for x, y in glob),
                        "tracks": ";".join(f"{t.track_id}:{t.x:.3f},{t.y:.3f},{int(t.confirmed)},{t.misses}"
                                           for t in self.tracker.tracks.values())})

    def _markers(self, stamp) -> None:
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        shown = self.tracker.tracks.values() if self.cfg.perception.show_rejected else self.obstacles
        obstacle_ids = {t.track_id for t in self.obstacles}
        for tr in shown:
            m = Marker()
            m.header.frame_id, m.header.stamp = self.odom_frame, stamp
            m.ns, m.id, m.type, m.action = "cones", tr.track_id, Marker.CYLINDER, Marker.ADD
            m.pose.position = Point(
                x=tr.x,
                y=tr.y,
                z=self.odom.ground_z() + self.cfg.perception.cone_height / 2,
            )
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = self.cfg.perception.cone_diameter
            m.scale.z = self.cfg.perception.cone_height
            is_obstacle = tr.track_id in obstacle_ids
            m.color.r, m.color.g, m.color.b = (1.0, 0.5, 0.0) if is_obstacle else (0.6, 0.6, 0.6)
            m.color.a = 0.9 if is_obstacle else 0.3
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def destroy_node(self):
        self.log.close()
        super().destroy_node()


def main(args=None):
    run_node(PerceptionNode)


if __name__ == "__main__":
    main()

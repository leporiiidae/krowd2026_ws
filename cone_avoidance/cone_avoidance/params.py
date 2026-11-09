"""Parameter defaults, loading, and derived values.

Every node and the offline simulator build the same :class:`Config` from the
same YAML (``config/cone_avoidance.yaml``), so a field tweak affects the real
vehicle and the simulator identically. Values marked TODO(실측) are placeholders
and no code path branches on their specific value.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict

import yaml


DEFAULTS: Dict[str, Any] = {
    "vehicle": {
        "length": 2.464,
        "wheelbase": 1.340,
        "front_overhang": 0.844,
        "rear_overhang": 0.280,        # 최후방은 뒷타이어
        "body_width": 1.160,           # TODO(실측)
        "track_width": 1.000,          # TODO(실측)
        "tire_width": 0.180,           # TODO(실측)
        "max_wheel_steer_deg": 21.0,   # 실측(2026-09-19) 기계적 끝단 (자전거모델 환산, 좌 25.0 / 우 21.5 중 보수값)
    },
    "frames": {
        # Front-axle centre expressed in the odometry child frame (base_link).
        "front_axle_x": 1.340,         # 실측(2026-09-19): base_link = 후륜축 (= 휠베이스)
        "front_axle_y": 0.0,
        # Ground height in base_link. The z band below is measured from here.
        "ground_z": -0.27,             # 실측(2026-09-19): base_link 가 지면보다 0.27 m 위 -> 지면은 -0.27
        "base_frame": "base_link",
        "use_tf": True,                # LiDAR -> base_link 를 TF 로 조회. false 면 아래 값 사용
        "lidar_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "steering": {
        "output_ratio": 6.8,           # 실측(2026-09-19): 엔코더 2.71 rad = 바퀴 21.5도, 2.75 rad = 바퀴 25.0도
        "output_limit_rad": 2.5,       # 펌웨어 ANGLE_LIMIT_RAD (2026-09-21: 1.7453 -> 2.5). 바퀴각 = 이 값/ratio
        "max_rate_deg_s": 17.0,        # 실측(2026-09-24, 펌웨어 스텝주기 상향 후) 바퀴 기준
        "rate_use_ratio": 0.7,
        "curvature_use_ratio": 0.8,
    },
    "reference": {
        # motion: 출발 후 실제로 이동한 방향으로 기준선 고정 (EKF 가 정지 중 yaw 를 못 믿는 차량용)
        # start_pose: 시작 명령 시점의 전륜축 위치/차량 yaw 로 기준선 고정
        # fixed: 아래 origin/heading 을 그대로 사용 (측량값이 있을 때)
        "mode": "start_pose",
        "recheck_distance": 3.0,       # start_pose: 이만큼 간 뒤 실제 이동 방향과 대조 (다르면 다시 고정)
        "motion_distance": 3.0,        # motion: 이만큼 직진한 뒤 이동 방향으로 고정
        "motion_distance_max": 8.0,    # yaw 가 끝내 안 맞아도 여기서는 고정 (플래그)
        "yaw_agree_deg": 10.0,         # 이동 방향과 EKF yaw 가 이 안으로 맞아야 고정
        "origin_x": 0.0,
        "origin_y": 0.0,
        "heading_deg": 0.0,
    },
    "rtk": {
        "enabled": True,
        "fix_status_min": 2,           # NavSatStatus.status >= 이 값이면 Fix 로 간주
        "timeout": 1.0,
    },
    "road": {
        "width": 3.8,                 # 차로 폭. 왼쪽 경계 = 중앙선, 오른쪽 경계 = 실선
        "lane_hard_constraint": True,
        # 중앙선(왼쪽)을 넘어도 감점이 없는 장애물 그룹 번호 (1 = 첫 번째). 실선(오른쪽)은 항상 금지
        "center_cross_groups": [2],
        "center_cross_extra": 1.0,     # 허용 시 차체가 중앙선을 넘어도 되는 최대 거리 (m)
        "lane_tolerance": 0.02,        # 차체 모서리의 차선 판정 허용 오차 (m)
    },
    "perception": {
        "detect_range": 15.0,
        "roi_x": [0.5, 20.0],
        "roi_y": [-3.0, 3.0],
        "ground_z": [0.10, 0.90],
        "cluster_eps": 0.25,
        "min_pts_at_5m": 15,
        "cone_diameter": 0.45,
        "cone_height": 0.80,
        "width_ratio_min": 0.5,        # 콘 지름 대비 클러스터 폭 하한 (±50%)
        "width_ratio_max": 1.5,
        "self_box": [-0.5, 2.7, 0.8],  # 자기 차체 제외 [x_min, x_max, |y|]. GPS 마스트가 x=2.5 m
        "tall_reject_margin": 0.20,    # 콘 높이보다 이만큼 더 높은 물체가 같은 자리에 있으면 콘 아님
        "top_check_range": 10.0,       # 이 거리 안에서는 콘 꼭대기 근처까지 점이 있어야 함
        "top_min_below": 0.30,         # 꼭대기가 (cone_height - 이 값) 보다 낮으면 콘 아님 (낮은 수풀)
        "row_pitch_ratio": 1.1,        # 콘 줄의 실제 피치 / 지름 (실측: 3개 줄이 1.52 m)
        "max_cones_per_cluster": 4,    # 붙어 있는 콘 열로 인정하는 최대 개수 (초과 = 벽/가드레일)
        "confirm_frames": 3,
        "miss_tolerance": 8,
        "track_filter_alpha": 0.4,
        "track_match_distance": 0.8,
        "pair_group_window": 1.5,
        "show_rejected": False,        # RViz 에 콘 줄로 인정 안 된 트랙도 회색으로 표시 (디버그)
        "deskew": True,
        "point_time_field": "t",
        "point_time_scale": 1.0e-9,    # Ouster t 는 ns
    },
    "planning": {
        "lateral_margin": 0.30,
        "plan_length_min": 4.0,
        "plan_length_max": 12.0,
        "plan_length_step": 0.5,
        "offset_step": 0.05,
        "offset_ladder_max": 6,        # 오프셋 확장 단계 수 상한 (10 Hz 계획 주기 안에 끝나도록)
        "return_length": 5.0,
        "clear_margin": 0.70,
        "transit_threshold_gap": 12.0,
        "sample_step": 0.1,
        "min_clearance": 0.0,          # 3원 근사 기준 채택 하한
        "clearance_bucket": 0.05,      # 이 이하의 여유 차이는 동률로 보고 완만한 경로 선택
        # Transit centre is searched around the body-aware window midpoint.
        "transit_center_offsets": [0.0, -0.25, 0.25, -0.5, 0.5],
        # Entry maneuver end is searched from the "bumper reaches cone" point
        # forward by this step until the front axle is abeam the cone.
        "entry_end_step": 0.25,
    },
    "replan": {
        "enabled": True,
        "period": 0.2,
        "pos_threshold": 0.15,
        "commit_distance": 2.0,
        "offset_filter_alpha": 0.3,
        "offset_tolerance": 0.03,      # 필터된 오프셋이 원시값과 이만큼 이상 다르면 재계획 계속
    },
    "control": {
        "rate_hz": 50.0,
        "k_stanley": 1.2,
        "k_soft": 0.5,
        "k_heading": 1.0,
        "latency_compensation": 0.60,  # 실측(2026-09-24): 순수지연 0.11 + 작은 조향 시상수 0.49
        # 조향 지령을 무슨 단위로 내보낼지. 외부 ros_to_can_node(/target_angle)는
        # 출력축 각도를 기대한다. wheel 이면 바퀴 각도 그대로 발행한다.
        "steer_cmd_unit": "output_shaft",   # output_shaft | wheel
        "speed_cmd_unit": "kmh",            # kmh | mps (외부 ros_to_can_node는 km/h)
        "steer_track_err_max_deg": 5.0,
        "steer_track_err_time": 0.5,
        "path_timeout": 0.5,
        "odom_timeout": 0.3,
    },
    "speed": {
        "target": 0.8333333333333334,  # 3 km/h
        "max_clamp": 1.67,             # 6 km/h
        "auto_from_geometry": True,
        "rtk_degraded": 0.5,
        "fault": 0.5,
    },
    "failure": {
        "never_stop": True,
        "fail_speed_step": 0.2,
        "fail_min_speed": 0.4,
    },
    "termination": {
        "pass_miss_frames": 5,
        # DONE: no cone seen for this many frames after the last pair.
        "done_miss_frames": 10,
        "done_sensor_mode": "either",  # either | both | yolo | lidar
    },
    "mission": {
        "auto_start": True,            # 런치 = 출발. false 로 두면 /cone_avoidance/start 를 기다린다
        "require_zero_done": False,    # true 면 영점 완료(0x202) 를 zero_wait_timeout 까지 기다린 뒤 출발
        "zero_wait_timeout": 3.0,
        "expected_pairs": 2,
        "pair_min_cones": 3,           # YOLO 개시 시 쌍에 확정돼 있어야 할 콘 수 (LiDAR 폴백은 1개로도 개시)
        "yolo_trigger_frames": 3,      # 회피 개시: YOLO 연속 검출 프레임
        "yolo_timeout": 0.5,           # 이 시간 이상 YOLO 메시지 없음 = YOLO 불가
        # YOLO 가 개시를 못 한 채 콘이 이만큼(전륜축 기준 필요거리 + 이 값) 가까워지면 LiDAR 로 개시
        "lidar_fallback_margin": 1.0,
        # 회피 개시 최소 거리. 기동이 들어갈 자리가 있으면 일찍 시작한다 (진입이 완만해지고,
        # 감속할 시간이 생기고, 확인할 여유가 있을 때 화면에 경로가 보인다)
        "trigger_distance_min": 12.0,
        # 규정: 그룹 i 가 붙어 있는 차선 경계. left = 중앙선, right = 실선. 통과는 반대쪽으로 한다.
        # 콘 줄의 바깥 끝을 그 경계로 보고 차선 중앙을 추정한다 (기준선 누적 오차와 무관해짐)
        "group_sides": ["left", "right"],
        "cones_per_group": 3,
        "row_min_cones": 2,            # 이 개수 이상 나란히 붙어 있어야 장애물 그룹 (먼 그룹은 2개만 보이기도 함)
        "row_max_cones": 4,            # 이보다 많이 이어진 무리는 콘 줄이 아님 (초과분은 잘라냄)
        "row_along_tol": 0.60,         # 한 줄의 콘들이 진행방향으로 벌어질 수 있는 최대 거리
        "row_angle_tol": 25.0,         # 줄 방향이 진행방향의 수직에서 벗어날 수 있는 각도 (deg)
        "row_clear_radius": 1.5,       # 콘 줄 주변에서 다른 물체를 세는 반경
        "row_lateral_max": 2.5,        # 줄의 안쪽 끝이 차량 진행선에서 이 이상 떨어지면 차선 밖 (가까운 거리 기준)
        "row_heading_tol": 10.0,       # 먼 줄일수록 차량 yaw 오차만큼 위 허용폭을 늘림 (deg)
        "row_clutter_min_partial": 2,  # 콘이 2개만 보이는 줄은 이웃 2개부터 수풀로 봄
        "row_clutter_min": 3,          # 그 안에 이 개수 이상 있으면 수풀 (1~2개로는 절대 지우지 않음)
        "row_dup_radius": 0.35,        # 이보다 가까운 트랙은 같은 콘의 중복으로 봄
        "row_gap_max": 0.80,           # 줄 안에서 이웃 콘 중심 간 최대 간격
        "row_len_slack": 0.60,         # 줄 길이 상한 = cones_per_group * 지름 + 이 값
        "lane_gate_loose": 1.5,        # 차선이 확정되기 전의 게이트 여유 (기준선 드리프트 허용)
        "lane_shift_max": 1.5,         # 첫 그룹이 가리키는 차선 중앙이 기준선에서 이 이상 벗어나면 잡음
        "lane_consistency": 0.8,       # 다음 그룹이 가리키는 차선 중앙이 이전과 이 이상 다르면 잡음
        "min_intrusion": 0.40,         # 차선 안으로 이만큼 이상 들어온 것만 장애물 그룹으로 센다
        "lane_gate_margin": 0.50,      # 차선 밖 이만큼까지의 콘만 인정 (길가 수풀 배제)
        "start_lateral_offset": 0.0,   # 출발 시 차량 d (차선 중앙 기준)
    },
    "topics": {
        "odometry": "/ekf/odometry",
        "pointcloud": "/ouster/points",
        "navsat": "/gps/fix",
        "yolo_count": "/yolo/cone_count",
        "start": "/cone_avoidance/start",
        "cones": "/cone_avoidance/cones",
        "raw_cones": "/cone_avoidance/raw_cones",
        "cone_markers": "/cone_avoidance/cone_markers",
        "mission_markers": "/cone_avoidance/mission_markers",
        "held_points": "/cone_avoidance/held_points",
        "lidar_raw_count": "/cone_avoidance/lidar_raw_count",
        "path": "/cone_avoidance/path",
        "speed_target": "/cone_avoidance/speed_target",
        "mission_status": "/cone_avoidance/mission_status",
        "steer_cmd": "/target_angle",
        "speed_cmd": "/target_speed",
        "control_status": "/cone_avoidance/control_status",
        "steer_feedback": "/cone_avoidance/steer_feedback",
        "wheel_speed": "/cone_avoidance/wheel_speed",
        "zero_done": "/cone_avoidance/zero_done",
    },
    "sim": {
        "dt": 0.02,
        # 기준선 좌표 (s, d) 를 평탄화: [s1, d1, s2, d2, ...]
        # 1차: 왼쪽 중앙선에 3개 밀착, 2차: 오른쪽 실선에 3개 밀착 (콘 지름 0.45, 간격 0)
        "cones": [10.0, 1.675, 10.0, 1.225, 10.0, 0.775,
                  15.0, -0.775, 15.0, -1.225, 15.0, -1.675],
        # 차량이 실제로 낼 수 있는 조향 각속도. 0 이면 steering.max_rate_deg_s 를 그대로 쓴다.
        # 계획기가 믿는 값과 다르게 두면 "따라갈 수 없는 경로" 상황을 재현할 수 있다.
        "actual_max_rate_deg_s": 0.0,
        "steer_delay": 0.11,           # 실측(2026-09-24 steer_probe) 순수 지연
        "steer_err_sat_rad": 0.147,    # 실측: 작은 조향의 시상수 0.49 s = sat/최대각속도
                                       # 작은 조향(3도)은 3.9 °/s 로 = 펌웨어 MAX_STEP_HALF_PERIOD_US(200us,
                                       # 1/64 마이크로스텝 2500 step/s) 의 최저 스텝속도에 묶여 있다
        "accel_limit": 0.8,            # m/s^2
        "pos_noise_std": 0.015,
        "yaw_noise_std_deg": 0.3,
        "odom_rate_hz": 50.0,
        "cone_noise_std_near": 0.03,
        "cone_noise_std_per_m": 0.006,
        "cone_detect_prob_near": 0.98,
        "cone_detect_prob_far": 0.6,
        "yolo_range": 12.0,
        "yolo_rate_hz": 8.0,
        "yolo_fov_deg": 70.0,
        "init_d_std": 0.05,
        "init_yaw_std_deg": 1.0,
        "course_length": 35.0,
    },
    "logging": {
        "enabled": True,
        "path": "./logs",
    },
}


class Section:
    """Attribute view over one nested parameter dict."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            setattr(self, key, Section(value) if isinstance(value, dict) else value)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def unflatten(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Turn ROS dotted names (``vehicle.wheelbase``) into a nested dict."""
    out: Dict[str, Any] = {}
    for name, value in flat.items():
        node = out
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


class Config:
    """Merged parameters plus the derived quantities every module needs."""

    def __init__(self, overrides: Dict[str, Any] | None = None):
        self.apply(overrides, reset=True)

    def apply(self, overrides: Dict[str, Any] | None = None, reset: bool = False) -> None:
        """(Re)build in place, so objects holding this Config see live parameter edits."""
        base = DEFAULTS if reset else self.raw
        self.raw = deep_merge(base, overrides or {})
        for key, value in self.raw.items():
            setattr(self, key, Section(value) if isinstance(value, dict) else value)

        v = self.vehicle
        # 최후방이 뒷타이어이므로 바퀴가 차체보다 튀어나오면 바퀴 바깥 폭이 충돌 폭.
        self.collision_width = max(v.body_width, v.track_width + v.tire_width)
        self.d_limit = self.road.width / 2.0 - self.collision_width / 2.0
        self.cone_radius = self.perception.cone_diameter / 2.0
        self.front_axle_to_rear = v.wheelbase + v.rear_overhang
        self.rear_axle_to_front = v.wheelbase + v.front_overhang
        if not 0.0 < v.max_wheel_steer_deg < 80.0:
            # tan(delta) is the curvature: it blows up at 90 deg and goes NEGATIVE beyond,
            # which would silently invert every steering limit. A road wheel never
            # reaches such an angle anyway.
            raise ValueError(
                f"vehicle.max_wheel_steer_deg={v.max_wheel_steer_deg} is not a usable road-wheel "
                "angle (must be 0-80 deg). If the steering output shaft turns further than the "
                "wheel, that difference belongs in steering.output_ratio.")
        # Usable steering is the tighter of the mechanical stop and the firmware soft
        # limit converted to a wheel angle. Planning against an angle the firmware will
        # clip would silently under-steer exactly when the path needs it most.
        mechanical = math.radians(v.max_wheel_steer_deg)
        ratio = self.steering.output_ratio
        self.max_wheel_steer_mechanical = mechanical
        self.max_wheel_steer_from_limit = (self.steering.output_limit_rad / ratio
                                           if ratio > 0 else math.inf)
        self.max_wheel_steer = min(mechanical, self.max_wheel_steer_from_limit)
        self.steer_limited_by = ("output_limit"
                                 if self.max_wheel_steer_from_limit < mechanical else "mechanical")
        self.max_steer_rate = math.radians(self.steering.max_rate_deg_s)
        # Planning budgets (margins against step loss / saturation).
        self.plan_steer_rate = self.steering.rate_use_ratio * self.max_steer_rate
        self.plan_max_steer = math.atan(
            self.steering.curvature_use_ratio * math.tan(self.max_wheel_steer)
        )
        self.kappa_limit = (
            self.steering.curvature_use_ratio * math.tan(self.max_wheel_steer) / v.wheelbase
        )

    @classmethod
    def from_yaml(cls, path: str, overrides: Dict[str, Any] | None = None) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        # Accept both the ROS 2 parameter layout and a plain nested dict.
        if "/**" in data:
            data = data["/**"].get("ros__parameters", {})
        return cls(deep_merge(data, overrides or {}))

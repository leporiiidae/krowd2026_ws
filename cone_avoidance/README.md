# cone_avoidance

직선 도로의 정적 콘 그룹 2개를 지그재그로 회피하는 ROS 2 (Humble, Python) 패키지.
`obstacle_avoidance_spec.md` 구현 + 2026-09-19 배치 변경 반영. 알고리즘은 전부 ROS 비의존 순수 파이썬이고, ROS 노드와
오프라인 시뮬레이터가 **같은 코드·같은 YAML**을 쓴다.

## 코스 배치 (2026-09-19 기준)

```
        ← 차로폭 3.8 m (왼쪽 = 중앙선, 오른쪽 = 실선) →
   ┆                                    │
   ┆                        ●●●         │  2차 그룹: 오른쪽 실선에 3개 밀착 → 왼쪽으로 통과
   ┆                                    │           (중앙선 넘어도 감점 없음)
   ┆              ↑ 4~6 m               │
   ┆●●●                                 │  1차 그룹: 왼쪽 중앙선에 3개 밀착 → 오른쪽으로 통과
   ┆                                    │           (중앙선 넘으면 안 됨 — 애초에 막혀 있음)
                 ↑ 진행 방향
```

- 한 그룹 = 지름 0.45 m 콘 3개가 간격 없이 가로로 붙은 줄 → 폭 1.35 m
- 목표 오프셋(마진 0.30): 1차 d = −0.34, 2차 d = +0.34 → swing 0.68 m. 차선 한계 d_limit = 1.31 m. 이 배치에선 실제로 중앙선을
  넘지 않지만, 2차 그룹이 더 안쪽으로 나와 있으면 `road.center_cross_extra`(기본 1.0 m)까지 넘어간다.
- 실선은 어느 그룹에서도 넘지 않는다. 계획기는 전륜축 오프셋뿐 아니라 **요각을 포함한 차체 네 모서리**가
  허용선 안에 있는지 검사한다(`road.lane_tolerance`).

## 구조

```
cone_avoidance/
  params.py        기본값 + 파생값(collision_width, d_limit ...) — YAML 과 1:1 (테스트로 강제)
  geometry.py      부호 규약, Frenet(직선), 5차 다항식, OffsetPath(구간 연결·C2 접합)
  swept.py         후축 견인 적분으로 차체 yaw 계산 → 3원/사각형 clearance, 조향 요구량
  planner.py       목표 오프셋, 진입/전이/복귀 탐색, 속도 자동 결정, 실패 사다리, 재계획
  mission.py       FSM + 재계획 규칙(커밋, 종료점 고정, 오프셋 필터)
  tracker.py       전역 좌표 콘 추적, 쌍 그룹핑
  lidar.py         ROI → 지면 제거 → 단일링크 클러스터 → 붙은 쌍 분리 → 원통 중심 피팅, 디스큐
  stanley.py       Stanley + 지연 보상
  can_codec.py     CAN 프레임 인코딩/디코딩
  sim.py           킥네마틱 시뮬레이터 + 몬테카를로
  *_node.py        ROS 노드 (perception / mission / control / can_bridge / sim_vehicle)
config/cone_avoidance.yaml
launch/cone_avoidance.launch.py   실차
launch/sim.launch.py              하드웨어 없이 전체 ROS 그래프 (합성 포인트클라우드)
launch/bag.launch.py              백 재생 + RViz (CAN 없음)
rviz/cone_avoidance.rviz
test/                             pytest 72개
```

| 노드 | 주기 | 입력 | 출력 |
|---|---|---|---|
| `perception` | LiDAR 프레임 | `/ouster/points`, `/ekf/odometry` | `cones`(확정 트랙), `raw_cones`, RViz 마커 |
| `mission` | 10 Hz | `cones`, `raw_cones`, `/yolo/cone_count`, `zero_done`, `start`, `/gps/fix` | `path`(기준선+5차 계수), `speed_target`, `mission_status`(JSON) |
| `control` | 50 Hz 독립 타이머 | `path`, `speed_target`, `/ekf/odometry`, `steer_feedback` | `steer_cmd`(바퀴 rad, 좌+), `speed_cmd`(m/s) |
| `can_bridge` | 50 Hz | `steer_cmd`, `speed_cmd` | CAN 0x110/0x120/0x130, `steer_feedback`, `wheel_speed`, `zero_done` |

YOLO 는 별도 프로세스이고 **`std_msgs/Int32` 콘 개수**만 넘기면 된다(`/yolo/cone_count`, 추론
프레임마다, 0 포함). 제어 루프와 프로세스가 달라 추론 지연이 제어 주기에 영향을 주지 않는다.

## 빌드 / 실행

```bash
cd ~/krowd2026_ws/cone_avoidance
source /opt/ros/humble/setup.bash
colcon build --packages-select cone_avoidance --symlink-install
source install/local_setup.bash

# 1) 오프라인 시뮬레이터 (ROS 불필요)
python3 -m cone_avoidance.sim --seeds 40                      # 몬테카를로
python3 -m cone_avoidance.sim --seeds 40 --gap 4 --set speed.target=1.5 --set steering.max_rate_deg_s=30
python3 -m cone_avoidance.sim --seed 3 --plot run.png --csv run.csv

# 2) ROS 그래프 전체를 가상 차량으로
ros2 launch cone_avoidance sim.launch.py

# 2-1) 기록된 백 재생 + RViz (CAN 송신 없음, 백 시계를 따름)
ros2 launch cone_avoidance bag.launch.py bag:=/mnt/c/Users/tk205/Downloads/0919_avoidance_f
#   paused:=true   일시정지로 시작 (재생 터미널에서 SPACE)   rate:=0.5  느리게
#   play:=false    백은 다른 터미널에서 직접: ros2 bag play <bag> --clock
#   use_rviz:=false  RViz 없이

# 3) 실차 — 런치하면 바로 주행한다 (mission.auto_start: true)
pip install python-can                                         # can_bridge 만 필요
ros2 launch cone_avoidance cone_avoidance.launch.py

#   차를 움직이지 않고 경로만 확인: use_can:=false (CAN 으로 아무것도 안 나감)
#   출발 명령을 받고 출발시키려면: mission.auto_start 를 false 로 두고
ros2 topic pub --once /cone_avoidance/start std_msgs/Bool "data: true"

# 조향 실측 (바퀴 들고. 속도 지령은 안 나간다) — max_rate_deg_s 와 latency_compensation 을 재 준다
ros2 run cone_avoidance steer_probe
ros2 run cone_avoidance steer_probe --ros-args -p amplitudes_deg:="[3.0, 6.0, 12.0, 18.0]" -p dwell:=2.0

# 주행 후 원인 분석 — 조향이 못 따라간 건지, 차가 경로를 벗어난 건지, 경로 자체가 틀린 건지
python3 -m cone_avoidance.check_log logs/control_*.csv

# 주행 중 튜닝 (재시작 불필요)
ros2 param set /cone_mission planning.lateral_margin 0.25
ros2 param set /cone_control control.k_stanley 1.5

# 테스트 (이 PC 는 anyio 플러그인이 깨져 있어 autoload 를 끈다)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
```

## 조향 실측 (2026-09-24, `steer_probe`)

가정했던 값이 크게 틀렸고, 그것이 "저속에선 피하는데 고속에선 박는다"의 원인이었다.

| | 가정 | 실측(펌웨어 수정 전) | 실측(MIN_STEP_HALF_PERIOD_US 하향 후) |
|---|---|---|---|
| 최대 각속도 (바퀴) | 60 °/s | 11.5 °/s | **17.2 °/s** |
| 작은 조향(3°) 속도 | — | 3.6 °/s | **3.9 °/s** |
| 순수 지연 | — | 0.12 s | 0.11 s |
| 작은 조향 시상수 | — | 0.52 s | **0.49 s** |
| 제어 지연 보상 | 0.15 s | — | **0.60 s** |

**두 개의 서로 다른 한계가 있다.**

- **최대 각속도**는 `MIN_STEP_HALF_PERIOD_US`(스텝 주파수 상한)가 정한다. 30 → 8 µs 로 3.75배를 기대했지만
  실제로는 11.5 → 17.2 °/s (1.5배)에 그쳤다. 모터·드라이버 토크가 먼저 한계다. 더 낮춰도 얻을 게 없다.
- **작은 조향 속도**는 `MAX_STEP_HALF_PERIOD_US`(200 µs = 2500 step/s, 1/64 마이크로스텝)가 정한다.
  3.9 °/s 가 이 최저 스텝속도와 정확히 일치한다. 미션에서 쓰는 조향은 대부분 작은 각이라
  **체감 반응은 전적으로 여기서 결정된다.** 스텝 주파수 상한을 아무리 올려도 이 값은 안 변한다.

지연 보상은 순수 지연이 아니라 **순수 지연 + 시상수**를 넣어야 한다. 시뮬레이터 스윕:

| 지연 보상 | 0.30 | 0.45 | **0.60** | 0.75 | 0.90 |
|---|---|---|---|---|---|
| 횡오차 max | 0.147 m | 0.118 m | **0.107 m** | 0.117 m | 0.119 m |

순수 지연만(0.12) 넣으면 횡오차가 0.76 m 로 커지고 `k_stanley` 를 올리면 발산한다.

**다음 개선은 `MAX_STEP_HALF_PERIOD_US` 200 → 100 µs** 다. 작은 조향이 2배 빨라져 시상수가
0.49 → 0.25 s, 횡오차가 0.107 → 0.076 m 로 줄어든다(시뮬레이터). 50 µs 까지 낮춰도 그 이상은 거의 없다.

**실차 첫 기동 전 확인:** 실측값(`steering.output_ratio` 6.8, `vehicle.max_wheel_steer_deg` 21,
`frames.front_axle_x` 1.34, `frames.ground_z` −0.27)은 2026-09-19 기준으로 들어가 있다.
남은 TODO 는 `vehicle.track_width`/`tire_width` 뿐이다
(`steering.max_rate_deg_s` 와 `control.latency_compensation` 은 2026-09-24 에 실측으로 확정).

**런치 = 출발. 트리거는 없다.** 출발 명령도, 영점 완료(`0x202`) 대기도 기본값에서는 없다
(`mission.auto_start: true`, `require_zero_done: false`). 외부에서 트리거를 걸려면 둘 중 하나:
`auto_start: false` 로 두고 `/cone_avoidance/start` 를 쏘거나, 런치 자체를 원하는 시점에 실행.

**CAN 은 외부에서 준비한다.** 노드는 인터페이스를 올리지 않고(`ip link set can0 up` 은 미리),
기동 시 영점 명령(`0x130`)도 보내지 않는다(`can.auto_zero_request: false`). 노드가 CAN 에 하는 일은
`0x110`/`0x120` 송신과 `0x200`/`0x201`/`0x202` 수신뿐이다. 차를 세워 둔 채 경로만 보려면 `use_can:=false`.

## 사양서와 다르게 한 부분 (결정 사항)

1. **Swept path 의 차체 yaw = 후축 견인 적분.** 전륜축이 경로를 따를 때 `dψ/dσ = sin(θ−ψ)/L` 을
   RK4 로 적분한다. 경로 접선각을 yaw 로 쓰면 4 m 기동에서 뒷타이어 위치가 10 cm 이상 어긋난다
   (`test_tangent_yaw_assumption_would_misplace_rear_tyre`). 같은 적분에서 요구 조향각
   `δ = θ − ψ` 가 나오므로 곡률·각속도 제약도 닫힌식이 아니라 실제 요구량으로 검사한다.
2. **속도 결정에 차체 길이 반영.** 7.4 닫힌식에 `L = gap − (1.62 + 0.844 + 2·0.225)` 를 넣으면
   gap 4 m 에서 0.02 m/s 가 나와 쓸 수 없다. 대신 후보 경로(길이 × 중심 위치)마다 위 적분으로
   ① 모든 콘과의 clearance ② m 당 조향 변화량 → 추종 가능한 최고 속도 를 구해 고른다.
   우선순위: 속도 버킷(0.2 m/s) → clearance 버킷(5 cm) → 가장 완만한 조향. 닫힌식 값은
   `v_allow_analytic` 으로 로그에만 남긴다. 10절의 "감속 후 재시도"가 이 탐색에 흡수돼 있다.
   - 참고: 4~6 m 기동은 휠베이스(1.34 m)와 비슷한 규모라 yaw 지연이 경로를 저역통과시켜
     실제 조향 요구가 `wheelbase·dκ/ds` 보다 훨씬 작다. 60 m 기동에서는 닫힌식과 3% 이내로 일치.
3. **`settle_margin` 삭제.**
4. **DONE / 개시.** 개시 = YOLO `yolo_trigger_frames` 연속 검출 + LiDAR 그룹 콘 `pair_min_cones`(3)개 확정.
   YOLO 가 죽었으면 LiDAR 거리로 폴백(`yolo_trigger_missed` 플래그). DONE = 쌍
   `expected_pairs` 개 처리 + 차선 중앙 복귀 완료 + `done_miss_frames` 동안 장애물 미검출
   (`done_sensor_mode: either` = YOLO 또는 LiDAR). DONE 이후에도 속도 지령은 유지된다.
5. **`steering.output_ratio = 6.8`, 사용 가능 조향각은 파생값** (2026-09-19 실측).

   | 방향 | 0x201 | 안쪽/바깥 바퀴 | 자전거모델 환산 | 비 |
   |---|---|---|---|---|
   | 우 | −2.71 rad | 22° / 21° | 21.5° | 7.23 |
   | 좌 | +2.75 rad | 29° / 22° | 25.0° | 6.29 |

   좌우 비가 13% 다르다. 단일 상수 6.8 을 쓴다 — Stanley 가 폐루프라 ±8% 게인 오차는 흡수된다.
   **우측 측정은 애커먼 기하와 맞지 않는다** (안쪽 22°면 바깥은 17° 여야 하는데 21° 로 쟀다).
   재측정 권장.

   사용 가능한 바퀴각 = `min(기계적 끝단, output_limit_rad / output_ratio)` 로 자동 계산한다.
   2026-09-21 펌웨어에서 `ANGLE_LIMIT_RAD` 가 1.7453 → **2.5**(출력축 143°) 로 올라가
   2.5 / 6.8 = 21.1° > 기계적 끝단 21° 이므로, 이제 한계는 기계가 정한다(`steer_limited_by: mechanical`).
   최소 회전반경 6.38 m → **4.36 m**. 펌웨어 값을 바꾸면 YAML `output_limit_rad` 도 같이 바꿀 것.
   90° 를 넘는 바퀴각은 `tan(δ)` 가 음수가 되어 조향 한계가 뒤집히므로 기동 시 거부한다.
6. **`angle_scale: 100` — 펌웨어 그대로 둔다.** 처음엔 1000 을 권했지만 그건 출력축과 바퀴가 1:1 이라는
   전제였다. 실제 변환비가 6.8 이라 100 에서도 CAN 한 칸 = **바퀴 0.084°**(5 m 주행 시 횡방향 1.4 cm)
   로 충분하다. 시뮬레이터에 0x120 인코딩/양자화를 넣고 40회 비교 — 100 과 1000 의 콘 여유·횡오차가
   동일했다. PC 와 펌웨어가 다르면 `can_bridge` 가 피드백/지령 비로 감지해 `angle_scale_mismatch`
   폴트를 올린다(감속). 펌웨어를 1000 으로 바꾼다면 YAML 도 같이 1000 으로.

그 외 구현 중 추가한 것:
- 밀착된 콘 3개는 한 클러스터로 나온다 → 줄 방향(주성분) 길이로 개수를 정하고 1-D k-means 로 분리.
  서로 가려 보이는 면이 적은 콘은 중심이 틀어질 수 있어, 양 끝 콘을 클러스터 끝점까지 밀어 **줄의 폭을
  과소평가하지 않게** 했다(계획기는 안쪽 끝 기준으로 피한다). `max_cones_per_cluster` 초과는 벽으로 버림.
- 각 기동의 clearance 는 **담당 그룹 콘만**으로 평가한다. 오프셋을 유지하는 구간이 다음 그룹과 겹쳐 보여
  불필요하게 오프셋을 키우던 문제를 수정.
- 보이는 면의 무게중심은 축에서 0.785 r 앞쪽 → 반지름 고정 원 피팅으로 중심 보정 (14 m 에서 6 cm 이내).
- 오프셋 저역통과는 재계획 때만 전진하므로, "필터값이 원시값과 3 cm 이상 다르면 재계획 계속" 조건 추가.
  콘이 쌍에 새로 합류하면(스텝 변화) 필터를 거치지 않고 즉시 반영.
- 확정된 트랙은 삭제하지 않는다(정적 콘 + RTK). 앞에서 놓치면 기존 추정 유지, 뒤로 가면 정상.
- `can.on_cmd_timeout: stop` — 제어 **프로세스가 죽은** 경우만 속도 0. 계획 실패·센서 폴트는 사양대로 감속만 한다.
  대회 규정상 이것도 안 되면 `hold` 로.

## 시뮬레이터 결과 (실측 조향 모델, 시드 40개)

| 간격 | 목표 | 충돌 | DONE | 콘 여유 min | 차선 위반 | 횡오차 max | 회피 구간 속도 |
|---|---|---|---|---|---|---|---|
| 4 m | 3 km/h | 0 | 40/40 | +0.209 m | 0 | 0.079 m | 2.4 km/h |
| 4 m | 5 km/h | 0 | 40/40 | +0.216 m | 0 | 0.112 m | 2.9 km/h |
| 5 m | 3 km/h | 0 | 40/40 | +0.235 m | 0 | 0.077 m | 3.0 km/h |
| 5 m | 5 km/h | 0 | 40/40 | +0.235 m | 0 | 0.107 m | 4.3 km/h |
| 6 m | 5 km/h | 0 | 40/40 | +0.251 m | 0 | 0.104 m | 4.5 km/h |

조향이 느려서 계획기가 회피 구간에서 스스로 2.4~4.5 km/h 로 낮춘다. 목표를 6 km/h 로 올려도 같다 —
튜닝이 아니라 물리적 한계이며, 계획기가 그 한계를 알고 있기 때문에 콘 여유는 유지된다.

액추에이터 모델: 순수 지연 60 ms + 오차 비례 각속도(0.125 rad 에서 포화 — 펌웨어 `MAX_RATE_ERROR_RAD`).
잡음: 측위 1.5 cm / 0.3°, 콘 3 cm + 0.6 cm/m, 15 m 검출률 60%, 초기 오차 5 cm / 1°.
실측 스텝응답(15절 3번)이 나오면 `sim.steer_delay`, `sim.steer_err_sat_rad`, `steering.max_rate_deg_s` 를 맞출 것.

### 기준선 방향이 가장 민감하다

`reference.mode: start_pose` 는 시작 시점의 차량 yaw 를 도로 방향으로 **믿는다**. 콘 회피 자체는 콘
상대 위치로 하므로 영향이 없지만, 차선 중앙·`d_limit` 이 그만큼 회전한다.

| 조건 (구 배치, gap 4 m, 시드 40) | 충돌 | 콘 여유 min | 도로 이탈 max |
|---|---|---|---|
| start_pose, 시작 yaw 오차 σ=1° | 0 | +0.176 m | 0 |
| start_pose, σ=3° | 0 | +0.111 m | **+1.59 m** |
| fixed (측량된 기준선), σ=3° | 0 | +0.156 m | 0 |

도로 방향을 RTK 로 두 점 찍어 `reference.mode: fixed` + `origin_x/y`, `heading_deg` 로 넣는 것을 권장.
start_pose 로 갈 거면 출발 정렬을 1° 이내로.

## RViz

`rviz/cone_avoidance.rviz`. Fixed Frame = `odom`.

| 표시 | 토픽 | 내용 |
|---|---|---|
| LiDAR (held while paused) | `/cone_avoidance/held_points` | 원본 클라우드. 백을 일시정지해도 2 Hz 로 마지막 스캔을 유지 (인식·제어는 이 토픽을 안 씀) |
| Cones (tracked) | `/cone_avoidance/cone_markers` | 주황 = 확정 트랙, 회색 = 후보 |
| Lanes / Vehicle / State | `/cone_avoidance/mission_markers` | 노랑 = 중앙선, 흰색 = 실선, 파랑 = 충돌 판정 사각형, 글자 = FSM 상태·속도·오프셋 |
| Planned Path | `/cone_avoidance/path_viz` | 전륜축 계획 경로 |

WSL 에서는 GPU OpenGL 경로에서 RViz 가 약 10 초 뒤 세그폴트한다(레퍼런스 `stop_mission.rviz` 도 동일).
런치가 WSL 을 감지해 `LIBGL_ALWAYS_SOFTWARE=1` 로 띄운다(`rviz_software_gl:=false` 로 끔).
백을 `/mnt/c/...` 에서 직접 재생하면 읽기가 느리다 — 부드럽게 보려면 `~/` 로 복사해서 재생.

백 재생용 덮어쓰기는 `config/bag_replay.yaml` (GPS 토픽 `/ublox_gps_node/fix`, 자동 출발, 영점 대기 없음).

## 실제 백(0919_avoidance_f)에서 찾아 고친 것

시뮬레이터에서는 안 보였던 문제들이다.

1. **EKF yaw 가 정지 중에 145° 틀려 있다가 출발 후 재정렬된다.** `start_pose` 기준선이 엉뚱한 방향이 되어
   → 새 기본값 `reference.mode: motion`: 출발 후 3 m 실제 이동 방향으로 기준선을 고정(EKF yaw 와 10° 안으로
   맞을 때). 그 3 m 동안은 직진.
2. **콘 구간이 출발점에서 33~38 m.** 기준선 방향 1° 오차가 거기선 0.6 m. 실제로 1차 그룹이 기준선 중앙에
   걸쳐 보여 계획이 **왼쪽(중앙선 너머)으로** 나갔다.
   → 규정상 1차는 중앙선, 2차는 실선에 붙어 있으므로(`mission.group_sides`) **콘 줄 바깥 끝 = 차선 경계**로
   보고 차선 중앙을 그룹에서 직접 추정한다. 통과 방향도 규정으로 고정. 실측: 기준선 대비 차선 중앙 −0.98 m.
3. **길가 수풀/구조물이 콘으로 잡혀 확정 트랙이 52개.**
   → LiDAR: 같은 자리에 콘보다 높은 게 있으면 제외, 10 m 안에서 너무 낮으면 제외, 자기 차체(GPS 마스트
   x = 2.5 m) 제외. 계획: 가로로 붙은 2개 이상의 콘 "줄"만 장애물 그룹, 차선 안으로 0.4 m 이상 들어온 것만.
4. **실제 콘 줄은 1.52 m** (3 × 0.45 = 1.35 가 아님, 틈 약 8 cm). → 분리 피치 `row_pitch_ratio: 1.1`.
5. **계획 1회에 1.8 초 걸린 적이 있다** (중앙선 넘기가 허용된 넓은 범위를 5 cm 간격으로 전부 탐색).
   → 후보 전체를 한 번에 벡터 연산(결과는 기존과 동일), 오프셋 단계 수 상한. 최악 189 → 42 ms.
   백 재생 시 계획 주기 99 백분위 0.103 s.

6. **길가 수풀이 "콘 줄"처럼 보이는 경우.** 인식 단계에서 장애물 후보를 콘 줄로만 제한한다
   (`tracker.find_rows`, `RowFilter`). 확정 트랙 27개 → 장애물 6개(= 콘 6개), 잡동사니 0개(모든 프레임).

   | 조건 | 진짜 콘 줄 (실측) | 수풀 (실측) |
   |---|---|---|
   | 가로로 붙은 콘 수 | 2~3 (먼 그룹은 2개만 보이기도 함) | — |
   | 도로에 대한 각도 | 수직에서 2~8° | 33~41° (일부 5°) |
   | 주변 1.5 m 안 다른 물체 | 0개 | 4~5개 |
   | 차량 진행선에서 거리 | 차선 안 | 4~5 m 밖 |

   **진짜 그룹을 지우는 쪽으로는 절대 틀리지 않게** 짰다:
   - 줄 바깥 끝 너머(갓길·반대 차선)의 물체는 세지 않는다. 2차 그룹은 실선에 붙어 있어 갓길 풀이 정상이다.
   - 물체 1~2개로는 줄을 지우지 않는다. 차로 위 물체는 그 자체가 장애물이므로, 줄에 붙어 있으면 장애물에 포함한다.
   - 한 번 인정된 줄은 계속 유지한다(콘은 움직이지 않는다).
   - 콘 2개만 보여도 인정한다. 3개로 제한했더니 먼 1차 그룹이 빠져 2차를 1차로 착각, 반대로 피해 차선을
     2 m 벗어났다(시뮬레이터). 그룹이 규정과 다른 쪽에 있으면 그 번호로 쓰지 않고 기다린다.
   - RViz 에서 걸러진 것까지 보려면 `perception.show_rejected: true` (회색).

결과(백 재생): 1차 그룹 오른쪽, 2차 그룹 왼쪽 통과로 계획, DONE 도달.
사람이 운전한 궤적은 2차 그룹 옆에서 차선 중앙 +1.7~1.9 m (중앙선 부근)까지 갔다.

## 로그

`logging.path` 아래에 실행마다 `control_*.csv`(50 Hz: pose, Frenet, FSM, 목표 오프셋 원시/필터, 계획 L,
`e_y`, heading 오차, 조향 지령 바퀴각/출력축각/CAN raw, `0x201` 실측, 추종 오차, 속도 지령/실측,
swept clearance 두 값, 폴트), `plan_*.csv`(10 Hz: 계획 이벤트, `v_allow`, `v_allow_analytic`),
`cones_*.csv`(LiDAR 프레임마다 원시 검출 + 트랙 상태).

## 메인 컴퓨터로 옮길 때

패키지 폴더만 복사 → `colcon build`. 의존성은 ROS Humble 기본 + `numpy`, `scipy`, `pyyaml`,
(`can_bridge` 만) `python-can`. `topics.*` 를 그쪽 토픽명에 맞추고, 기존 `ros_to_can_node` /
`can_to_ros_node` 와 **동시에 띄우지 말 것**(같은 CAN ID 로 송신한다). 기존 브리지를 계속 쓰려면
`use_can:=false` 로 띄우고 `steer_cmd` × `output_ratio` 를 `/target_angle` 로 넘기는 어댑터가 필요하다.

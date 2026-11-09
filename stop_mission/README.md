# stop_mission

Standalone cone-course mission adapted from `xycar_ws/src/cone_mode` for the KROWD
vehicle and the `stop_mission_{test01,run01,run02}` ROS 2 bags.

## Bag topic mapping

| Role | xycar source | stop-mission bag / this package |
|---|---|---|
| LiDAR | `/scan` (`LaserScan`) | `/ouster/points` (`PointCloud2`) |
| measured speed | `/xycar_motor` | `/ekf/odometry.twist` (`Odometry`) |
| cloud extrinsic | assumed `base_link` | bag `/tf_static`: `os_lidar` -> `base_link` |
| cone boundary | `/cone_markers` | `/stop_mission/cones` |
| cone search fans | none | `/stop_mission/search_sectors` (`MarkerArray`) |
| center path | `/cone_path` | `/stop_mission/path` |
| PP lookahead target | none | `/stop_mission/lookahead_marker` (red sphere) |
| last cone distance | none | `/stop_mission/end_distance` (`Float32`, m) |
| steering output | `XycarMotor.angle` | `/target_angle` (`Float32`, rad) |
| speed output | `XycarMotor.speed` | `/target_speed` (`Float32`, m/s) |
| safety stop | zero `XycarMotor` | `/brake` (`Bool`) |

The perception node removes the ground by a configurable base-link Z ROI, voxelizes
the cloud, performs unordered Euclidean clustering, filters compact cone-sized
clusters, and classifies blue(left)/yellow(right) lines. The cluster-size thresholds
were measured on the run01/run02 bags (32-ring Ouster, 1.37-degree ring spacing): a
cone gives about 16 voxel points at 2-5 m, 11 at 5-8 m, 6 at 8-11 m, 5 at 11-14 m,
and only a single ring beyond 14 m, so `min_cluster_points: 5` keeps cones within
about 11 m and `min_cluster_height` cannot usefully exceed 0.20 m. The planner uses the midpoint
of opposite-color Delaunay edges, with paired/one-side fallbacks. The controller is a
standalone Pure Pursuit controller using the KROWD 1.34 m wheelbase.
The controller uses a 3.0 m lookahead: it walks the ordered forward path and selects the
first point whose Euclidean distance from `base_link` is at least 3.0 m. If the path is
shorter, it uses the available forward point closest to 3.0 m. The exact point used in
the steering calculation is published as a red 0.5 m sphere on
`/stop_mission/lookahead_marker`; it is deleted whenever tracking stops or braking begins.

Perception can be launched before any bag/live sensor publisher. It waits for the first
PointCloud2 that can be transformed into `base_link`, then starts classification. Each
cloud supplies 3D cone candidates, while the initialized left/right XY chains persist
between frames. Initial seed/bootstrap classification runs only until both sides have
at least two cones; later clouds extend the compensated chain tails through the
configured 17.5-degree (half-angle) forward sectors. Perception uses the full `/ekf/odometry` pose:
successive position and quaternion-yaw samples form the relative SE(2) transform that
moves retained chains into the current `base_link` frame. Candidates less
than 1.5 m ahead or within 1.0 m of the vehicle centerline cannot seed a side. Chains
have no age timeout; points are removed after passing behind `x=-0.5 m` or when the
40-point bound is exceeded.
The retained chains are used only for side association and forward-sector search.
`/stop_mission/cones`, which is consumed directly by the path planner as well as RViz,
contains only current-frame cone centroids from `/stop_mission/raw_cones` with their
blue/yellow side labels applied. Thus a historical chain shifted by an EKF yaw correction
cannot become a path-planning coordinate. The compensated historical coordinates remain
available separately on `/stop_mission/chain_markers` for debugging. Current detections
are first associated to the nearest same-side chain; after two consecutive association-
loss frames, perception rebuilds both lines from current LiDAR coordinates only when at
least two plausible cones are visible on each side.
Because this straight mission's EKF can realign its initial heading after the vehicle
begins moving, a yaw change of 5 degrees or more between LiDAR frames is treated as an
EKF frame correction rather than physical steering. That delta is not applied to
retained cones. Perception rebuilds both sides from the current gray-cylinder
coordinates on every frame and returns to persistent EKF-compensated tracking after
five consecutive frames with at most 1 degree of yaw change. A backward odometry
timestamp also clears the full EKF sample buffer, so replaying a bag without restarting
the launch behaves like its first playback.
Synthetic opposite-side cones are disabled because the planner already supports a
one-side offset path; this prevents near-field clutter from creating a false boundary.
`/ekf/odometry.twist` remains a controller input for final stop confirmation. A
backward sensor timestamp, such as replaying a bag from the beginning
without restarting the launch, resets perception, planner, and controller replay state
automatically.

The driving target is configured with `target_speed_mps` in
`config/stop_mission.yaml` and defaults to 4.0 m/s (14.4 km/h). The controller
publishes it to `/target_speed`, whose `ros_can_bridge` interface contract remains
m/s. Mission braking is independent of
vehicle speed: `/stop_mission/end_distance` must be at or below 3.0 m for ten
consecutive 20 Hz control frames while the path and endpoint remain fresh. A larger,
stale, or invalid distance resets the confirmation counter. Once confirmed, braking
latches and the outputs switch directly to zero speed and brake. Measured speed is used
only to confirm the final stop.
When the final tracked cone enters that distance, braking is latched and the outputs
switch directly to zero speed and brake. `/stop_mission/done` is published after
`/ekf/odometry` confirms that the vehicle has stopped.

## Build and bag replay

```bash
cd ~/krowd_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select stop_mission
source install/local_setup.bash

# Terminal 1
ros2 launch stop_mission stop_mission.launch.py use_sim_time:=true

# Terminal 2: prepare the player in a paused state.
ros2 bag play ~/stop_mission_run01 \
  --clock \
  --start-paused \
  --topics /ouster/points /ekf/odometry /tf /tf_static

```

After all nodes are ready, press `Space` in Terminal 2 to start or pause playback.
`--rate 0.5` may be added while inspecting the result in RViz.
The main launch starts the dedicated RViz2 window by default. Pass
`use_rviz:=false` only for a headless run. `stop_mission_rviz.launch.py` remains
available when RViz must be started separately.

`/stop_mission/held_points` relays every live `/ouster/points` message immediately so
the gray RViz cloud and classified markers show the same sensor frame. Pausing rosbag
stops new sensor messages and simulation time, so a wall-clock timer additionally
repeats the most recently received scan at 2 Hz while paused. This topic is for RViz
only and is never consumed by perception or control. Before the bag has emitted its
first scan there is nothing to repeat, so a player started paused must be resumed at
least once before a held cloud can appear.

The dedicated RViz config enables the held LiDAR cloud, classified blue/yellow cone
markers, search-sector outlines, and the Pure Pursuit path. Cone and sector markers
keep stable IDs; each LiDAR frame updates them and individually deletes only IDs that
disappeared instead of clearing every marker before repainting the frame.
Each fan starts at the last classified pair on that side and its line shows the exact
configured search radius and half-angle used to extend the line. Its Fixed Frame is `base_link`;
raw gray cone candidates and Delaunay debug markers are omitted. The saved view
controller is `Orbit`, so drag in the 3D view to rotate between top, side, and
perspective views; use the mouse wheel to zoom.

`auto_start` is true for bag testing. For a mission manager, set it false and publish
`std_msgs/Bool(data=true)` on `/stop_mission/start`. The Pure Pursuit node publishes
directly to the existing `ros_can_bridge` input topics, so do not run another controller
that publishes `/target_angle`, `/target_speed`, or `/brake` at the same time.

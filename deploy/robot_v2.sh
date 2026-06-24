#!/usr/bin/env bash
# v2 launcher — uses pipeline_v2 with OSNet + pose + body proportions.
# Mirror of ~/robot.sh but runs the v2 perception script.
#
# Run v1 (original):     bash ~/robot.sh
# Run v2 (this file):    bash ~/robot_v2.sh
#
# v2 is HEADLESS (terminal output only). For the full Flask UI in v2,
# see migration notes at the bottom of run_perception_hailo_v2.py.
set -u
LOGDIR=/tmp
ROS_ENV='source /opt/ros/jazzy/setup.bash && export TURTLEBOT3_MODEL=burger ROS_DOMAIN_ID=30 LDS_MODEL=LDS-02 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
HAILO_ENV='source ~/hailo_env/bin/activate'

declare -A PIDS=()
declare -a ORDER=(bringup relay safety lidarscan lidarbridge perception)

start_one() {
  local name="$1" cmd="$2" delay="$3"
  : > "$LOGDIR/${name}_v2.log"
  bash -c "$cmd" >>"$LOGDIR/${name}_v2.log" 2>&1 &
  PIDS[$name]=$!
  sleep "$delay"
}

start_all() {
  echo "Starting v2 services..."
  start_one bringup    "$ROS_ENV && ros2 launch turtlebot3_bringup robot.launch.py"     6
  start_one relay      "$ROS_ENV && /usr/bin/python3.12 ~/cmd_vel_relay.py"             2
  start_one safety     "$ROS_ENV && /usr/bin/python3.12 ~/safety_node.py"               2
  # This lidar (LD14-class) talks the LDROBOT 0x54 protocol at 115200, but the stock
  # ld08_driver opens it at 230400 and gets only noise. Kill that driver and run our
  # serial->/scan node at the correct baud instead. (bracket pattern avoids self-kill)
  start_one lidarscan  "pkill -f '[l]d08_driver'; sleep 1; $ROS_ENV && /usr/bin/python3.12 ~/lidar_serial_scan.py" 2
  start_one lidarbridge "$ROS_ENV && /usr/bin/python3.12 ~/lidar_bridge.py"             2
  start_one perception "$HAILO_ENV && cd ~/capstone-robot-follow && USE_HAILO=1 python scripts/run_perception_hailo_v2.py" 4
}

stop_all() {
  ( bash -c "$ROS_ENV && for i in 1 2 3; do
      ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped '{}' >/dev/null 2>&1
      sleep 0.1
    done" ) &
  for name in perception safety lidarbridge lidarscan relay bringup; do
    pid="${PIDS[$name]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      pkill -P "$pid" 2>/dev/null
      kill "$pid" 2>/dev/null
    fi
  done
  sleep 1
  pkill -f run_perception_hailo_v2 2>/dev/null
  pkill -f cmd_vel_relay           2>/dev/null
  pkill -f safety_node             2>/dev/null
  pkill -f lidar_bridge            2>/dev/null
  pkill -f lidar_serial_scan       2>/dev/null
  pkill -f ld08_driver             2>/dev/null
  pkill -f robot.launch            2>/dev/null
  PIDS=()
}

board() {
  clear
  IP=$(hostname -I | awk '{print $1}')
  echo "══════════════════════════════════════════"
  echo "  TurtleBot3 Robot — V2 (OSNet + pose + proportions)"
  echo "  Headless mode — see perception_v2.log for output"
  echo "══════════════════════════════════════════"
  for name in "${ORDER[@]}"; do
    pid="${PIDS[$name]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      printf "  ● %-12s PID %s\n" "$name" "$pid"
    else
      printf "  ○ %-12s (down)\n" "$name"
    fi
  done
  echo "──────────────────────────────────────────"
  echo "  [s] stop all   [r] restart"
  echo "  [l] tail perception log   [q] quit"
  echo "──────────────────────────────────────────"
}

trap 'echo; echo "Shutting down..."; stop_all; exit 0' INT TERM EXIT

start_all
while :; do
  board
  read -t 2 -n 1 key
  case "${key:-}" in
    s) stop_all ;;
    r) stop_all; start_all ;;
    l) clear; echo "Ctrl+C to return"; tail -F "$LOGDIR/perception_v2.log" ;;
    q) break ;;
  esac
done

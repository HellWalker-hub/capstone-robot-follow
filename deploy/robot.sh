#!/usr/bin/env bash
# Single controller for all robot services.
#   bash ~/robot.sh
# Keys: s=stop, r=restart, l=tail perception log, q=quit
set -u
LOGDIR=/tmp
ROS_ENV='source /opt/ros/jazzy/setup.bash && export TURTLEBOT3_MODEL=burger ROS_DOMAIN_ID=30 LDS_MODEL=LDS-02 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
HAILO_ENV='source ~/hailo_env/bin/activate'

declare -A PIDS=()
declare -a ORDER=(bringup relay safety lidar perception)

start_one() {
  local name="$1" cmd="$2" delay="$3"
  : > "$LOGDIR/$name.log"
  bash -c "$cmd" >>"$LOGDIR/$name.log" 2>&1 &
  PIDS[$name]=$!
  sleep "$delay"
}

start_all() {
  echo "Starting services..."
  start_one bringup    "$ROS_ENV && ros2 launch turtlebot3_bringup robot.launch.py"     6
  start_one relay      "$ROS_ENV && /usr/bin/python3.12 ~/cmd_vel_relay.py"             2
  start_one safety     "$ROS_ENV && /usr/bin/python3.12 ~/safety_node.py"               2
  start_one lidar      "$ROS_ENV && /usr/bin/python3.12 ~/lidar_bridge.py"              2
  start_one perception "$HAILO_ENV && cd ~/capstone-robot-follow && USE_HAILO=1 python scripts/run_perception_hailo.py" 4
}

stop_all() {
  # send zero twist a few times as a safety
  ( bash -c "$ROS_ENV && for i in 1 2 3; do
      ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped '{}' >/dev/null 2>&1
      sleep 0.1
    done" ) &
  for name in perception safety lidar relay bringup; do
    pid="${PIDS[$name]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      pkill -P "$pid" 2>/dev/null
      kill "$pid" 2>/dev/null
    fi
  done
  sleep 1
  # nuke any survivors by pattern
  pkill -f run_perception_hailo 2>/dev/null
  pkill -f cmd_vel_relay        2>/dev/null
  pkill -f safety_node          2>/dev/null
  pkill -f lidar_bridge         2>/dev/null
  pkill -f robot.launch         2>/dev/null
  PIDS=()
}

board() {
  clear
  IP=$(hostname -I | awk '{print $1}')
  echo "══════════════════════════════════════════"
  echo "  TurtleBot3 Robot — running"
  echo "  Open http://${IP}:5000/ in your browser"
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
    l) clear; echo "Ctrl+C to return"; tail -F "$LOGDIR/perception.log" ;;
    q) break ;;
  esac
done

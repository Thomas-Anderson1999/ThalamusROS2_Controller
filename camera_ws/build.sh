if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

VENV="$HOME/ros_workplace/venv"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

colcon build

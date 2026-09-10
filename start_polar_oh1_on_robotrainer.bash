#!/bin/bash

# The Polar OHA should only connect to the USB Dongle
USB_BLUETOOTH_MAC="BC:FC:E7:21:3B:E8"
DEFAULT_BLUETOOTH_MAC="CC:2F:71:F8:25:FE"

# Check if desired controller is visible
if timeout 2 bluetoothctl <<< $'list\nexit\n' | grep -q "$USB_BLUETOOTH_MAC"; then
    echo "USB Bluetooth Dongle ($USB_BLUETOOTH_MAC) found."
else
    echo -e "\033[1;31mError: USB Bluetooth Dongle is not visible\nTry to plug it in and out!\033[0m"
    exit 1
fi

# Disable the unwanted Bluetooth controller so that it remains off by default.
# echo -e "select ${DEFAULT_BLUETOOTH_MAC}\npower off\nexit" | bluetoothctl

# Select the desired controller and ensure it's powered on
# echo -e "select ${USB_BLUETOOTH_MAC}\npower on\nexit" | bluetoothctl

# Reset terminal settings in case bluetoothctl has modified them
stty sane

# Check if the desired USB Bluetooth controller is UP RUNNING
DESIRED_STATUS=$(hciconfig -a | awk -v mac="$USB_BLUETOOTH_MAC" 'BEGIN {IGNORECASE=1} 
    $0 ~ mac {found=1} 
    found && /UP RUNNING/ { print; exit }')

if [ -z "$DESIRED_STATUS" ]; then
    echo -e "\033[1;31mError: USB Bluetooth Dongle ($USB_BLUETOOTH_MAC) is not UP RUNNING!\033[0m"
    exit 1
else
    echo "USB Bluetooth Dongle is UP RUNNING."
fi

# Start the Polar OH1 container in the foreground
# cd "$HOME/workspace/docker/robotrainer_docker_polar_oh1/" && ./detached.sh
./start_docker.sh "/bin/bash -c 'ros2 launch polar_oh1 ros2_polar_oh1.launch.py'"

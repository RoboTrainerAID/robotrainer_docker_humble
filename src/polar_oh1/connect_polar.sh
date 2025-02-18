#!/bin/bash

echo "🔍 Start Bluetooth..."
sudo service bluetooth start

# MAC Address of Polar OH1+ (Replace with your device's address)
DEVICE_MAC="A0:9E:1A:E0:BC:97"

for i in {1..10}; do
    echo "🔍 Attempt $i: Scanning for Polar OH1+..."
    bluetoothctl <<EOF
    power on
    agent on
    default-agent
    scan on
EOF

    # Wait for the scan to complete
    sleep 10  # Adjust the sleep time if needed

    echo "🔗 Attempt $i: Connecting to Polar OH1+..."
    bluetoothctl <<EOF
    scan off
    pair $DEVICE_MAC
EOF
    sleep 10

    bluetoothctl <<EOF
    trust $DEVICE_MAC
EOF
    sleep 10

    bluetoothctl <<EOF
    connect $DEVICE_MAC
EOF
    sleep 10

    # Check if the device is connected
    if bluetoothctl info $DEVICE_MAC | grep -q "Connected: yes"; then
        echo "✅ Connected to Polar OH1+!"
        exit 0
    fi

    echo "❌ Attempt $i: Failed to connect to Polar OH1+."
done

echo "❌ Failed to connect to Polar OH1+ after 10 attempts."
exit 1
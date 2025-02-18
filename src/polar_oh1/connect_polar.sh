#!/bin/bash

# MAC Address of Polar OH1+ (Replace with your device's address)
DEVICE_MAC="A0:9E:1A:E0:BC:97"

echo "🔍 Scanning for Polar OH1+..."
bluetoothctl <<EOF
power on
agent on
default-agent
scan on
EOF

# Wait for the scan to complete
sleep 10  # Adjust the sleep time if needed

echo "🔗 Connecting to Polar OH1+..."
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

echo "✅ Connected to Polar OH1+!"
exit
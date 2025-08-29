#!/bin/bash
# Quick Robot Connection Test
# Usage: ./test_robot_connection.sh

echo "=== Robot Connection Test ==="

echo "Scanning network for devices..."
DEVICE_COUNT=$(nmap -sn 192.168.1.0/24 | grep -c "Host is up")
echo "Found $DEVICE_COUNT devices on 192.168.1.0/24 network"

if [ "$DEVICE_COUNT" -ge 2 ]; then
    echo "✓ Robot connection GOOD! ($DEVICE_COUNT devices detected)"
    echo ""
    echo "Devices found:"
    nmap -sn 192.168.1.0/24 | grep "Nmap scan report"
    
    # Try to ping the robot directly
    echo ""
    echo "Testing direct robot ping..."
    if ping -c 1 192.168.1.2 >/dev/null 2>&1; then
        echo "✓ Robot responds to ping at 192.168.1.2"
    else
        echo "? Robot doesn't respond to ping (might be at different IP)"
        echo "Other devices on network:"
        nmap -sn 192.168.1.0/24 | grep -v "$(hostname)" | grep "Nmap scan report"
    fi
    
else
    echo "✗ Robot NOT DETECTED (only $DEVICE_COUNT device found)"
    echo "Expected: 2 devices (computer + robot)"
    echo ""
    echo "Current device:"
    nmap -sn 192.168.1.0/24 | grep "Nmap scan report"
    echo ""
    echo "Try running reset scripts:"
    echo "  sudo ./reset_usb.sh       # Reset USB subsystem"
    echo "  sudo ./reset_network.sh   # Reset network interface"
fi

echo ""
echo "=== Test Complete ==="

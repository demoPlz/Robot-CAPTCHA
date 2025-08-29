#!/bin/bash
# Network Reset Script for Robot Connection
# Usage: sudo ./reset_network.sh

echo "=== Network Reset Script for Robot Connection ==="

# Check current network status
echo "1. Current network status:"
ip addr show | grep -E "(eth|wlan|enp)" | head -5

# Reset network interface
echo "2. Resetting network interfaces..."

# Find the primary ethernet interface
ETH_INTERFACE=$(ip route | grep default | head -1 | sed 's/.*dev \([^ ]*\).*/\1/')
echo "Primary interface: $ETH_INTERFACE"

if [ ! -z "$ETH_INTERFACE" ]; then
    echo "Resetting $ETH_INTERFACE..."
    sudo ip link set $ETH_INTERFACE down
    sleep 2
    sudo ip link set $ETH_INTERFACE up
    sleep 3
    
    # Renew DHCP if available
    sudo dhclient -r $ETH_INTERFACE 2>/dev/null
    sudo dhclient $ETH_INTERFACE 2>/dev/null
fi

# Restart NetworkManager (if available)
if systemctl is-active --quiet NetworkManager; then
    echo "3. Restarting NetworkManager..."
    sudo systemctl restart NetworkManager
    sleep 5
fi

# Test robot connection
echo "4. Testing robot connection..."
echo "Scanning network for devices..."
DEVICE_COUNT=$(nmap -sn 192.168.1.0/24 | grep -c "Host is up")
echo "Found $DEVICE_COUNT devices on network"

if [ "$DEVICE_COUNT" -ge 2 ]; then
    echo "✓ Robot connection restored! ($DEVICE_COUNT devices detected)"
    echo "Devices found:"
    nmap -sn 192.168.1.0/24 | grep "Nmap scan report"
else
    echo "✗ Robot still not detected (only $DEVICE_COUNT device found)"
    echo "Expected: 2 devices (computer + robot)"
fi

echo "=== Network Reset Complete ==="

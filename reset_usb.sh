#!/bin/bash
# USB Reset Script - Run this when USB subsystem crashes
# Usage: sudo ./reset_usb.sh

echo "=== USB Reset Script ==="
echo "This will reset the USB subsystem without rebooting"

# Method 1: Reload USB controller modules
echo "1. Reloading USB controller modules..."
sudo modprobe -r xhci_pci xhci_hcd ehci_pci ehci_hcd ohci_pci ohci_hcd uhci_hcd
sleep 2
sudo modprobe xhci_pci xhci_hcd ehci_pci ehci_hcd ohci_pci ohci_hcd uhci_hcd
sleep 3

# Method 2: Reset USB controllers via sysfs
echo "2. Resetting USB controllers via sysfs..."
for usb_ctrl in /sys/bus/pci/drivers/[eoux]hci_hcd/*/remove; do
    if [ -w "$usb_ctrl" ]; then
        echo "Removing $(dirname $usb_ctrl)"
        sudo echo 1 > "$usb_ctrl"
    fi
done

sleep 2

# Rescan PCI bus to re-detect USB controllers
echo "3. Rescanning PCI bus..."
sudo echo 1 > /sys/bus/pci/rescan

sleep 3

# Method 3: Reset individual USB devices/hubs (optional)
echo "4. Resetting USB hubs..."
for hub in /sys/bus/usb/devices/*/authorized; do
    if [ -w "$hub" ]; then
        sudo echo 0 > "$hub"
        sleep 0.5
        sudo echo 1 > "$hub"
    fi
done

echo "5. Checking USB and network status..."
lsusb | wc -l
echo "USB devices detected: $(lsusb | wc -l)"

echo "6. Testing robot network connection..."
DEVICE_COUNT=$(nmap -sn 192.168.1.0/24 | grep -c "Host is up")
echo "Network devices found: $DEVICE_COUNT"

if [ "$DEVICE_COUNT" -ge 2 ]; then
    echo "✓ Robot connection restored! ($DEVICE_COUNT devices on network)"
else
    echo "✗ Robot still not detected (only $DEVICE_COUNT device found)"
    echo "You may need to run: sudo ./reset_network.sh"
fi

echo "=== USB Reset Complete ==="
echo "Run './test_robot_connection.sh' to verify robot connection"

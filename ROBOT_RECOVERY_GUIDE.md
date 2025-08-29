# Robot Connection Recovery Guide

When your robot connection is lost after a USB subsystem crash, try these steps:

## Quick Test
```bash
./test_robot_connection.sh
```
This will scan for devices on 192.168.1.0/24. You should see 2 devices when robot is connected.

## Recovery Methods (try in order)

### 1. USB Reset (Most Common Fix)
```bash
sudo ./reset_usb.sh
```
This reloads USB controller modules and rescans the PCI bus.

### 2. Network Reset (If robot uses network)
```bash
sudo ./reset_network.sh
```
This resets network interfaces and restarts NetworkManager.

### 3. Manual USB Module Reset
```bash
# Quick single command
sudo modprobe -r xhci_pci && sudo modprobe xhci_pci

# Full reset
sudo modprobe -r xhci_pci xhci_hcd ehci_pci ehci_hcd ohci_pci ohci_hcd uhci_hcd
sudo modprobe xhci_pci xhci_hcd ehci_pci ehci_hcd ohci_pci ohci_hcd uhci_hcd
```

### 4. Check System Logs
```bash
# Check for USB/hardware errors
sudo dmesg | tail -20
sudo journalctl -u NetworkManager --since "5 minutes ago"
```

## Expected Results
- **Before recovery**: `nmap -sn 192.168.1.0/24` shows 1 device (your computer)
- **After recovery**: `nmap -sn 192.168.1.0/24` shows 2 devices (computer + robot)

## If Nothing Works
- Check robot power/ethernet cables
- Try different USB ports
- As last resort: restart computer

## Prevention
Add this to your robot control script for better cleanup:
```python
import signal
import atexit

def cleanup_handler(signum, frame):
    # Proper robot disconnect
    robot.disconnect()
    # Camera cleanup
    crowd_interface.cleanup_cameras()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)
```

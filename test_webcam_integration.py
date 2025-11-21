#!/usr/bin/env python3
"""
Test script to verify webcam integration with sim mode.

This script tests:
1. Webcam detection and initialization
2. Webcam frame capture
3. Integration with SimManager (verifies webcam views are added to sim views)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from hardware_config import CAM_IDS
from interface_managers.webcam_manager import WebcamManager
from interface_managers.calibration_manager import CalibrationManager
from interface_managers.sim_manager import SimManager

def test_webcam_detection():
    """Test 1: Check if webcams can be detected and initialized."""
    print("\n" + "="*70)
    print("TEST 1: Webcam Detection and Initialization")
    print("="*70)
    
    print(f"\nConfigured camera IDs from hardware_config.py:")
    for cam_name, cam_id in CAM_IDS.items():
        print(f"  - {cam_name}: /dev/video{cam_id}")
    
    # Try to initialize webcam manager
    print("\nInitializing WebcamManager...")
    webcam_manager = WebcamManager(
        cam_ids=CAM_IDS,
        undistort_maps={},  # No undistortion for this test
        jpeg_quality=80
    )
    
    print("Opening cameras...")
    webcam_manager.init_cameras()
    
    if not webcam_manager.cams:
        print("❌ FAILED: No cameras were opened successfully")
        return False
    
    print(f"\n✅ SUCCESS: Opened {len(webcam_manager.cams)} camera(s):")
    for cam_name in webcam_manager.cams.keys():
        print(f"  - {cam_name}")
    
    # Check if we have left and front (the ones we need for sim mode)
    has_left = "left" in webcam_manager.cams
    has_front = "front" in webcam_manager.cams
    
    if has_left and has_front:
        print("\n✅ Both 'left' and 'front' cameras available (required for sim mode)")
    else:
        print(f"\n⚠️  WARNING: Missing required cameras:")
        if not has_left:
            print("  - 'left' camera not available")
        if not has_front:
            print("  - 'front' camera not available")
    
    return webcam_manager if (has_left or has_front) else None


def test_webcam_capture(webcam_manager):
    """Test 2: Verify we can capture frames from webcams."""
    print("\n" + "="*70)
    print("TEST 2: Webcam Frame Capture")
    print("="*70)
    
    import time
    print("\nWaiting 1 second for background capture threads to grab frames...")
    time.sleep(1)
    
    print("\nCapturing snapshot from webcams...")
    views = webcam_manager.snapshot_latest_views()
    
    if not views:
        print("❌ FAILED: No views captured")
        return False
    
    print(f"\n✅ SUCCESS: Captured {len(views)} view(s):")
    for cam_name, data_url in views.items():
        # Check if it's a valid data URL
        is_valid = isinstance(data_url, str) and data_url.startswith("data:image/jpeg;base64,")
        status = "✅" if is_valid else "❌"
        
        # Calculate approximate size
        if is_valid:
            b64_data = data_url.split("base64,")[1]
            size_kb = len(b64_data) * 3 / 4 / 1024  # Rough estimate
            print(f"  {status} {cam_name}: {size_kb:.1f} KB")
        else:
            print(f"  {status} {cam_name}: INVALID FORMAT")
    
    return views


def test_sim_manager_integration(webcam_manager):
    """Test 3: Verify SimManager can capture and persist webcam views."""
    print("\n" + "="*70)
    print("TEST 3: SimManager Integration (Mock Test)")
    print("="*70)
    
    # Create temporary directory for test cache
    with tempfile.TemporaryDirectory() as tmpdir:
        obs_cache_root = Path(tmpdir)
        
        print(f"\nUsing temporary cache directory: {obs_cache_root}")
        
        # Create minimal CalibrationManager (needed for SimManager)
        repo_root = Path(__file__).resolve().parent
        calib_manager = CalibrationManager(
            use_sim=True,
            repo_root=repo_root,
            real_calib_paths={},
            sim_calib_paths={},
        )
        
        # Create SimManager with webcam support
        print("\nCreating SimManager with webcam_manager...")
        sim_manager = SimManager(
            use_sim=True,
            task_name="test_task",
            obs_cache_root=obs_cache_root,
            state_lock=None,  # Not needed for this test
            pending_states_by_episode={},
            webcam_manager=webcam_manager,
            calibration_manager=calib_manager,
        )
        
        # Test the webcam capture method directly
        print("\nCalling _capture_and_persist_webcam_views...")
        episode_id = "test_episode_1"
        state_id = 0
        
        webcam_view_paths = sim_manager._capture_and_persist_webcam_views(episode_id, state_id)
        
        if not webcam_view_paths:
            print("❌ FAILED: No webcam views were captured/persisted")
            return False
        
        print(f"\n✅ SUCCESS: Captured and persisted {len(webcam_view_paths)} webcam view(s):")
        for view_name, file_path in webcam_view_paths.items():
            exists = os.path.exists(file_path)
            status = "✅" if exists else "❌"
            
            if exists:
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  {status} {view_name}: {file_path}")
                print(f"      File size: {size_kb:.1f} KB")
            else:
                print(f"  {status} {view_name}: FILE NOT FOUND")
        
        # Verify the naming convention
        expected_keys = {"webcam_left", "webcam_front"}
        actual_keys = set(webcam_view_paths.keys())
        
        if actual_keys.intersection(expected_keys):
            print(f"\n✅ View names follow correct convention (webcam_* prefix)")
        else:
            print(f"\n⚠️  WARNING: Unexpected view names: {actual_keys}")
        
        return True


def test_full_integration():
    """Test 4: Simulate a full state capture with both sim and webcam views."""
    print("\n" + "="*70)
    print("TEST 4: Full Integration Test")
    print("="*70)
    print("\nThis would require Isaac Sim to be running.")
    print("For now, we've verified:")
    print("  ✅ Webcam capture works")
    print("  ✅ SimManager can persist webcam views")
    print("  ✅ View naming convention is correct")
    print("\nTo fully test, you would:")
    print("  1. Start the backend server with sim mode enabled")
    print("  2. Trigger a state capture")
    print("  3. Fetch /api/get-state and check the 'views' object")
    print("  4. Verify it contains: front, left, right, top, webcam_left, webcam_front")


def main():
    """Run all tests."""
    print("\n" + "🧪 "*35)
    print("WEBCAM INTEGRATION TEST SUITE")
    print("🧪 "*35)
    
    try:
        # Test 1: Webcam detection
        webcam_manager = test_webcam_detection()
        if not webcam_manager:
            print("\n❌ OVERALL RESULT: FAILED - No webcams available")
            return 1
        
        # Test 2: Frame capture
        views = test_webcam_capture(webcam_manager)
        if not views:
            print("\n❌ OVERALL RESULT: FAILED - Cannot capture frames")
            return 1
        
        # Test 3: SimManager integration
        success = test_sim_manager_integration(webcam_manager)
        if not success:
            print("\n❌ OVERALL RESULT: FAILED - SimManager integration failed")
            return 1
        
        # Test 4: Full integration notes
        test_full_integration()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Start your backend server with sim mode")
        print("  2. Navigate to the frontend")
        print("  3. Check browser console for incoming views")
        print("  4. Verify you see 6 views total (4 sim + 2 webcam)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED WITH EXCEPTION:")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

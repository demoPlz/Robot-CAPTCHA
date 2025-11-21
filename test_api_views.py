#!/usr/bin/env python3
"""
Quick test to verify webcam views are being sent via the API.

Usage:
    python test_api_views.py [API_URL]

Example:
    python test_api_views.py https://ztclab-1.tail503d36.ts.net
"""

import json
import sys
import requests

def test_api_state_endpoint(api_url):
    """Fetch /api/get-state and check for webcam views."""
    
    print("\n" + "="*70)
    print("Testing API Endpoint for Webcam Views")
    print("="*70)
    
    endpoint = f"{api_url}/api/get-state"
    print(f"\nFetching: {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we got a state or a status response
        if "status" in data and "views" not in data:
            print(f"\n⚠️  Received status response: {data.get('status')}")
            if "message" in data:
                print(f"   Message: {data.get('message')}")
            print("\n   This is normal if no states are ready yet.")
            print("   Try triggering a state capture on the robot side.")
            return False
        
        # Check for views in the response
        views = data.get("views", {})
        
        if not views:
            print("\n❌ FAILED: No 'views' field in response")
            print(f"\nResponse keys: {list(data.keys())}")
            return False
        
        print(f"\n✅ SUCCESS: Found {len(views)} view(s) in response:")
        
        # Categorize views
        sim_views = []
        webcam_views = []
        other_views = []
        
        for view_name in views.keys():
            if view_name.startswith("webcam_"):
                webcam_views.append(view_name)
            elif view_name in ["front", "left", "right", "top"]:
                sim_views.append(view_name)
            else:
                other_views.append(view_name)
        
        if sim_views:
            print(f"\n  Sim views ({len(sim_views)}):")
            for v in sorted(sim_views):
                print(f"    ✅ {v}")
        
        if webcam_views:
            print(f"\n  Webcam views ({len(webcam_views)}):")
            for v in sorted(webcam_views):
                print(f"    🎥 {v}")
        
        if other_views:
            print(f"\n  Other views ({len(other_views)}):")
            for v in sorted(other_views):
                print(f"    📷 {v}")
        
        # Verify we have the expected webcam views
        expected_webcam = {"webcam_left", "webcam_front"}
        actual_webcam = set(webcam_views)
        
        if expected_webcam.issubset(actual_webcam):
            print(f"\n✅ PERFECT: Both expected webcam views are present!")
        elif actual_webcam:
            print(f"\n⚠️  PARTIAL: Found webcam views but not all expected ones")
            print(f"   Expected: {expected_webcam}")
            print(f"   Got: {actual_webcam}")
        else:
            print(f"\n❌ FAILED: No webcam views found (expected {expected_webcam})")
        
        # Check camera models and poses
        camera_models = data.get("camera_models", {})
        camera_poses = data.get("camera_poses", {})
        
        print(f"\n📐 Camera models: {len(camera_models)} defined")
        print(f"📍 Camera poses: {len(camera_poses)} defined")
        
        # Additional info
        if "state_id" in data:
            print(f"\n📝 State ID: {data['state_id']}")
        if "episode_id" in data:
            print(f"📁 Episode ID: {data['episode_id']}")
        
        return len(webcam_views) > 0
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ FAILED: Cannot connect to {api_url}")
        print("   Make sure the backend server is running.")
        return False
    except requests.exceptions.Timeout:
        print(f"\n❌ FAILED: Request timeout")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ FAILED: HTTP Error {e.response.status_code}")
        print(f"   {e}")
        return False
    except json.JSONDecodeError:
        print(f"\n❌ FAILED: Invalid JSON response")
        print(f"   Response: {response.text[:200]}")
        return False
    except Exception as e:
        print(f"\n❌ FAILED: Unexpected error")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the API test."""
    
    # Default to the Tailscale URL from the HTML file
    default_url = "https://ztclab-1.tail503d36.ts.net"
    
    if len(sys.argv) > 1:
        api_url = sys.argv[1].rstrip("/")
    else:
        api_url = default_url
        print(f"Using default API URL: {api_url}")
        print(f"(Pass a different URL as argument to override)")
    
    success = test_api_state_endpoint(api_url)
    
    print("\n" + "="*70)
    if success:
        print("✅ TEST PASSED: Webcam views are being sent via API!")
        print("="*70)
        print("\nYou can now check the frontend to see if the views are displayed.")
        return 0
    else:
        print("⚠️  TEST INCOMPLETE: See details above")
        print("="*70)
        print("\nPossible reasons:")
        print("  - Server is not running")
        print("  - No states are ready yet (try triggering a state capture)")
        print("  - Webcams are not connected or failed to initialize")
        return 1


if __name__ == "__main__":
    sys.exit(main())

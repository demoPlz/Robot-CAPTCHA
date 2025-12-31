#!/usr/bin/env python3
"""Cleanup script to delete all MTurk HITs in sandbox and production.

Run this to clean up before restarting to avoid duplicate HITs.
"""

import argparse
import boto3
from botocore.exceptions import ClientError


def cleanup_hits(sandbox=True):
    """Delete all HITs in MTurk sandbox or production.
    
    Args:
        sandbox: If True, clean sandbox. If False, clean production.
    """
    env_name = "sandbox" if sandbox else "production"
    endpoint_url = (
        "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
        if sandbox
        else "https://mturk-requester.us-east-1.amazonaws.com"
    )
    
    # Connect to MTurk
    mturk = boto3.client(
        "mturk",
        endpoint_url=endpoint_url,
        region_name="us-east-1",
    )
    
    try:
        # Get all HITs
        print(f"Finding all HITs in {env_name}...")
        response = mturk.list_hits(MaxResults=100)
        hits = response.get("HITs", [])

        if not hits:
            print(f"No HITs found in {env_name}")
            return

        print(f"Found {len(hits)} HIT(s)")

        # Delete each HIT
        for hit in hits:
            hit_id = hit["HITId"]
            hit_status = hit["HITStatus"]
            
            try:
                # First expire the HIT if it's still active
                if hit_status in ["Assignable", "Unassignable"]:
                    print(f"  Expiring HIT {hit_id}...")
                    mturk.update_expiration_for_hit(
                        HITId=hit_id,
                        ExpireAt=0  # Expire immediately
                    )
                
                # Then delete it
                print(f"  Deleting HIT {hit_id}...")
                mturk.delete_hit(HITId=hit_id)
                print(f"  Deleted HIT {hit_id}")
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "RequestError" and "has already been disposed" in str(e):
                    print(f"  HIT {hit_id} already deleted")
                else:
                    print(f"  Failed to delete HIT {hit_id}: {e}")

        print(f"\nCleanup complete for {env_name}!")

    except ClientError as e:
        print(f"Error listing HITs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup MTurk HITs")
    parser.add_argument(
        "--sandbox-only",
        action="store_true",
        help="Clean sandbox only (default: clean both)"
    )
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="Clean production only (default: clean both)"
    )
    args = parser.parse_args()
    
    print("MTurk Cleanup Script")
    print("=" * 50)
    
    if args.sandbox_only:
        print("\nCleaning SANDBOX:")
        cleanup_hits(sandbox=True)
    elif args.production_only:
        print("\nCleaning PRODUCTION:")
        cleanup_hits(sandbox=False)
    else:
        # Default: clean both
        print("\nCleaning SANDBOX:")
        cleanup_hits(sandbox=True)
        print("\n" + "=" * 50)
        print("\nCleaning PRODUCTION:")
        cleanup_hits(sandbox=False)

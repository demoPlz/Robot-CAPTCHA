#!/usr/bin/env python3
"""Cleanup script to delete all MTurk HITs in sandbox.

Run this to clean up before restarting to avoid duplicate HITs.
"""

import boto3
from botocore.exceptions import ClientError


def cleanup_sandbox_hits():
    """Delete all HITs in MTurk sandbox."""
    # Connect to sandbox
    mturk = boto3.client(
        "mturk",
        endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com",
        region_name="us-east-1",
    )

    try:
        # Get all HITs
        print("🔍 Finding all HITs in sandbox...")
        response = mturk.list_hits(MaxResults=100)
        hits = response.get("HITs", [])

        if not hits:
            print("✅ No HITs found in sandbox")
            return

        print(f"📋 Found {len(hits)} HIT(s)")

        # Delete each HIT
        for hit in hits:
            hit_id = hit["HITId"]
            hit_status = hit["HITStatus"]
            
            try:
                # First expire the HIT if it's still active
                if hit_status in ["Assignable", "Unassignable"]:
                    print(f"⏸️  Expiring HIT {hit_id}...")
                    mturk.update_expiration_for_hit(
                        HITId=hit_id,
                        ExpireAt=0  # Expire immediately
                    )
                
                # Then delete it
                print(f"🗑️  Deleting HIT {hit_id}...")
                mturk.delete_hit(HITId=hit_id)
                print(f"✅ Deleted HIT {hit_id}")
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "RequestError" and "has already been disposed" in str(e):
                    print(f"⚠️  HIT {hit_id} already deleted")
                else:
                    print(f"❌ Failed to delete HIT {hit_id}: {e}")

        print(f"\n✅ Cleanup complete!")

    except ClientError as e:
        print(f"❌ Error listing HITs: {e}")


if __name__ == "__main__":
    print("🧹 MTurk Sandbox Cleanup")
    print("=" * 50)
    cleanup_sandbox_hits()

#!/usr/bin/env python3
"""Approve all submitted assignments and cleanup MTurk HITs.

This combined script:
1. First approves all submitted assignments (pays workers)
2. Then deletes all HITs to clean up

Run this when you're done with a session to pay workers and clean up.
"""

import argparse
import boto3
from botocore.exceptions import ClientError


def approve_and_cleanup(sandbox: bool = False):
    """Approve all submitted assignments then delete all HITs.
    
    Args:
        sandbox: Whether to use sandbox environment
    """
    endpoint_url = (
        "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
        if sandbox
        else "https://mturk-requester.us-east-1.amazonaws.com"
    )
    
    client = boto3.client(
        "mturk",
        endpoint_url=endpoint_url,
        region_name="us-east-1",
    )
    
    env = "SANDBOX" if sandbox else "PRODUCTION"
    print(f"\n{'=' * 60}")
    print(f"MTurk Cleanup & Payment - {env}")
    print(f"{'=' * 60}\n")
    
    # ========== STEP 1: APPROVE ALL SUBMITTED ASSIGNMENTS ==========
    print(f"🔍 STEP 1: Checking for submitted assignments...\n")
    
    try:
        paginator = client.get_paginator('list_hits')
        hits = []
        for page in paginator.paginate():
            hits.extend(page['HITs'])
        
        if not hits:
            print(f"No HITs found in {env}")
            return
        
        print(f"Found {len(hits)} HIT(s)\n")
        
        total_approved = 0
        total_failed = 0
        
        # Approve submitted assignments
        for hit in hits:
            hit_id = hit['HITId']
            hit_status = hit['HITStatus']
            
            try:
                response = client.list_assignments_for_hit(
                    HITId=hit_id,
                    AssignmentStatuses=['Submitted'],
                )
                
                submitted = response.get('Assignments', [])
                
                if submitted:
                    print(f"📋 HIT {hit_id} ({hit_status}):")
                    print(f"   Found {len(submitted)} submitted assignment(s)")
                    
                    for assignment in submitted:
                        assignment_id = assignment['AssignmentId']
                        worker_id = assignment['WorkerId']
                        
                        try:
                            client.approve_assignment(
                                AssignmentId=assignment_id,
                                RequesterFeedback="Thank you for your contribution!",
                            )
                            print(f"   ✅ Approved assignment {assignment_id} from worker {worker_id}")
                            total_approved += 1
                            
                        except ClientError as e:
                            error_msg = str(e)
                            if "already been approved" in error_msg:
                                print(f"   ⏭️  Assignment {assignment_id} already approved")
                            else:
                                print(f"   ❌ Failed to approve {assignment_id}: {e}")
                                total_failed += 1
                    
                    print()
                    
            except ClientError as e:
                print(f"⚠️  Failed to check HIT {hit_id}: {e}\n")
        
        # Approval summary
        print("Approval Summary:")
        print(f"  ✅ Approved: {total_approved}")
        if total_failed > 0:
            print(f"  ❌ Failed: {total_failed}")
        print()
        
    except ClientError as e:
        print(f"❌ Error during approval: {e}\n")
    
    # ========== STEP 2: DELETE ALL HITS ==========
    print(f"🧹 STEP 2: Cleaning up HITs...\n")
    
    try:
        # Refresh HIT list
        response = client.list_hits(MaxResults=100)
        hits = response.get("HITs", [])

        if not hits:
            print(f"No HITs to clean up")
            return

        deleted_count = 0
        failed_count = 0
        
        for hit in hits:
            hit_id = hit["HITId"]
            hit_status = hit["HITStatus"]
            
            try:
                # First expire the HIT if it's still active
                if hit_status in ["Assignable", "Unassignable"]:
                    client.update_expiration_for_hit(
                        HITId=hit_id,
                        ExpireAt=0  # Expire immediately
                    )
                
                # Then delete it
                client.delete_hit(HITId=hit_id)
                print(f"  ✅ Deleted HIT {hit_id}")
                deleted_count += 1
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "RequestError" and "has already been disposed" in str(e):
                    print(f"  ⏭️  HIT {hit_id} already deleted")
                else:
                    print(f"  ❌ Failed to delete HIT {hit_id}: {e}")
                    failed_count += 1

        # Cleanup summary
        print()
        print("Cleanup Summary:")
        print(f"  ✅ Deleted: {deleted_count}")
        if failed_count > 0:
            print(f"  ❌ Failed: {failed_count}")

    except ClientError as e:
        print(f"❌ Error during cleanup: {e}")
    
    # ========== FINAL SUMMARY ==========
    print(f"\n{'=' * 60}")
    print(f"✅ Complete! Approved {total_approved} assignment(s) and deleted {deleted_count} HIT(s)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Approve all submitted assignments and cleanup MTurk HITs"
    )
    parser.add_argument(
        "--sandbox-only",
        action="store_true",
        help="Process sandbox only (default: process both)"
    )
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="Process production only (default: process both)"
    )
    args = parser.parse_args()
    
    if args.sandbox_only:
        approve_and_cleanup(sandbox=True)
    elif args.production_only:
        approve_and_cleanup(sandbox=False)
    else:
        # Default: process both
        print("\n🔄 Processing SANDBOX first...")
        approve_and_cleanup(sandbox=True)
        print("\n🔄 Processing PRODUCTION...")
        approve_and_cleanup(sandbox=False)

#!/usr/bin/env python3
"""Approve all submitted MTurk assignments.

This script finds all assignments in 'Submitted' status and approves them.
Useful when the main program exits before auto-approval completes.
"""

import argparse
import boto3
from botocore.exceptions import ClientError


def approve_all_assignments(sandbox: bool = False):
    """Approve all submitted assignments in MTurk.
    
    Args:
        sandbox: Whether to use sandbox environment
    """
    # Setup MTurk client
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
    print(f"\n🔍 Checking for submitted assignments in {env}...\n")
    
    # Get all HITs
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
        
        # Check each HIT for submitted assignments
        for hit in hits:
            hit_id = hit['HITId']
            hit_status = hit['HITStatus']
            
            try:
                # Get all submitted assignments for this HIT
                response = client.list_assignments_for_hit(
                    HITId=hit_id,
                    AssignmentStatuses=['Submitted'],
                )
                
                submitted = response.get('Assignments', [])
                
                if submitted:
                    print(f"📋 HIT {hit_id} ({hit_status}):")
                    print(f"   Found {len(submitted)} submitted assignment(s)")
                    
                    # Approve each submitted assignment
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
        
        # Summary
        print("=" * 60)
        print(f"✅ Approved: {total_approved}")
        if total_failed > 0:
            print(f"❌ Failed: {total_failed}")
        print("=" * 60)
        
    except ClientError as e:
        print(f"❌ Error accessing MTurk: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Approve all submitted MTurk assignments"
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use MTurk sandbox environment instead of production",
    )
    
    args = parser.parse_args()
    
    if not args.sandbox:
        print("⚠️  WARNING: This will approve assignments in PRODUCTION")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled")
            return
    
    approve_all_assignments(sandbox=args.sandbox)


if __name__ == "__main__":
    main()

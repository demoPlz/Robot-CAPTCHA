#!/usr/bin/env python3
"""Force cleanup a stuck HIT by expiring and disposing.

For HITs stuck in 'Unassignable' state with pending assignments.
"""

import argparse
import boto3
import time
from botocore.exceptions import ClientError


def force_cleanup_hit(hit_id: str, sandbox: bool = False):
    """Force cleanup a HIT by expiring and waiting for it to become deletable.
    
    Args:
        hit_id: HIT ID to cleanup
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
    print(f"\n🔧 Force cleanup HIT {hit_id} in {env}\n")
    
    try:
        # Get HIT status
        hit = client.get_hit(HITId=hit_id)['HIT']
        status = hit['HITStatus']
        pending = hit['NumberOfAssignmentsPending']
        completed = hit['NumberOfAssignmentsCompleted']
        
        print(f"Current status: {status}")
        print(f"Pending: {pending}, Completed: {completed}\n")
        
        # Step 1: Expire the HIT
        print("1️⃣ Expiring HIT...")
        try:
            client.update_expiration_for_hit(HITId=hit_id, ExpireAt=0)
            print("   ✅ Expired\n")
        except ClientError as e:
            if "already expired" in str(e).lower():
                print("   ⏭️  Already expired\n")
            else:
                raise
        
        # Step 2: Wait for assignments to clear
        if pending > 0:
            print(f"2️⃣ Waiting for {pending} pending assignment(s) to timeout...")
            print("   This can take 1-5 minutes after expiration")
            print("   Workers will be forced to return their assignments\n")
            
            for i in range(30):  # Wait up to 5 minutes
                time.sleep(10)
                hit = client.get_hit(HITId=hit_id)['HIT']
                new_pending = hit['NumberOfAssignmentsPending']
                new_status = hit['HITStatus']
                
                print(f"   [{i*10}s] Status: {new_status}, Pending: {new_pending}")
                
                if new_pending == 0 and new_status in ['Reviewable', 'Reviewing']:
                    print("\n   ✅ Assignments cleared!\n")
                    break
            else:
                print("\n   ⚠️  Timeout waiting for assignments to clear")
                print("   You may need to wait longer or contact MTurk support\n")
                return
        
        # Step 3: Try to delete
        print("3️⃣ Attempting to delete HIT...")
        try:
            client.delete_hit(HITId=hit_id)
            print("   ✅ HIT deleted successfully!\n")
        except ClientError as e:
            print(f"   ❌ Failed: {e}\n")
            
            # If still can't delete, try approving any remaining assignments
            print("4️⃣ Checking for submitted assignments to approve...")
            response = client.list_assignments_for_hit(
                HITId=hit_id,
                AssignmentStatuses=['Submitted']
            )
            
            if response.get('Assignments'):
                for assignment in response['Assignments']:
                    assignment_id = assignment['AssignmentId']
                    try:
                        client.approve_assignment(
                            AssignmentId=assignment_id,
                            RequesterFeedback="Auto-approved for cleanup"
                        )
                        print(f"   ✅ Approved {assignment_id}")
                    except ClientError:
                        pass
                
                print("\n5️⃣ Retrying delete after approval...")
                time.sleep(2)
                try:
                    client.delete_hit(HITId=hit_id)
                    print("   ✅ HIT deleted successfully!\n")
                except ClientError as e:
                    print(f"   ❌ Still failed: {e}")
                    print("\n⚠️  Manual cleanup required:")
                    print(f"   Visit: https://requester{'sandbox' if sandbox else ''}.mturk.com/manage")
            else:
                print("   No assignments to approve\n")
        
    except ClientError as e:
        print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Force cleanup a stuck MTurk HIT"
    )
    parser.add_argument(
        "hit_id",
        help="HIT ID to cleanup",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use MTurk sandbox environment",
    )
    
    args = parser.parse_args()
    force_cleanup_hit(args.hit_id, args.sandbox)


if __name__ == "__main__":
    main()

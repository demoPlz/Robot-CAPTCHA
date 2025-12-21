#!/usr/bin/env python3
"""Test script for MTurk sandbox setup.

This script tests the MTurk integration by:
1. Verifying AWS credentials are configured
2. Checking MTurk sandbox connectivity
3. Creating a test HIT
4. Listing the HIT
5. Cleaning up the test HIT

Run this script after configuring AWS credentials with `aws configure`.
"""

import sys
import time

import boto3
from botocore.exceptions import ClientError


def test_mturk_sandbox():
    """Test MTurk sandbox connectivity and HIT creation."""
    
    print("=" * 70)
    print("MTurk Sandbox Test Script")
    print("=" * 70)
    print()
    
    # Step 1: Initialize MTurk client for sandbox
    print("📡 Step 1: Connecting to MTurk sandbox...")
    try:
        mturk = boto3.client(
            'mturk',
            endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com',
            region_name='us-east-1'
        )
        print("✅ Connected to MTurk sandbox")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\nMake sure you have:")
        print("  1. Run 'aws configure' with your credentials")
        print("  2. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    # Step 2: Check account balance
    print("\n💰 Step 2: Checking account balance...")
    try:
        balance = mturk.get_account_balance()
        print(f"✅ Account balance: {balance['AvailableBalance']}")
        print(f"   Note: This is fake money in the sandbox")
    except ClientError as e:
        print(f"❌ Failed to get balance: {e}")
        return False
    
    # Step 3: Create a test HIT
    print("\n🎯 Step 3: Creating test HIT...")
    
    # Simple ExternalQuestion XML
    external_url = "https://www.example.com/test"  # Placeholder URL
    external_question = f"""
    <ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
        <ExternalURL>{external_url}</ExternalURL>
        <FrameHeight>600</FrameHeight>
    </ExternalQuestion>
    """
    
    try:
        response = mturk.create_hit(
            Title='Test HIT - Robot Control Task',
            Description='This is a test HIT for the robot control crowdsourcing system',
            Keywords='test, robot, sandbox',
            Reward='0.10',
            MaxAssignments=1,
            LifetimeInSeconds=300,  # 5 minutes
            AssignmentDurationInSeconds=300,
            AutoApprovalDelayInSeconds=60,
            Question=external_question,
        )
        
        hit_id = response['HIT']['HITId']
        hit_type_id = response['HIT']['HITTypeId']
        
        print(f"✅ Test HIT created successfully!")
        print(f"   HIT ID: {hit_id}")
        print(f"   HIT Type ID: {hit_type_id}")
        print(f"\n   Preview URL:")
        print(f"   https://workersandbox.mturk.com/mturk/preview?groupId={hit_type_id}")
        
    except ClientError as e:
        print(f"❌ Failed to create HIT: {e}")
        return False
    
    # Step 4: List HITs
    print("\n📋 Step 4: Listing your HITs...")
    try:
        hits = mturk.list_hits(MaxResults=10)
        print(f"✅ Found {hits['NumResults']} HIT(s) in your account")
        
        for hit in hits['HITs']:
            print(f"   - {hit['HITId']}: {hit['Title']} ({hit['HITStatus']})")
    
    except ClientError as e:
        print(f"❌ Failed to list HITs: {e}")
    
    # Step 5: Clean up test HIT
    print("\n🧹 Step 5: Cleaning up test HIT...")
    try:
        # Expire the HIT immediately
        mturk.update_expiration_for_hit(
            HITId=hit_id,
            ExpireAt=0
        )
        print(f"✅ HIT expired")
        
        # Wait a moment for the expiration to process
        time.sleep(2)
        
        # Delete the HIT
        mturk.delete_hit(HITId=hit_id)
        print(f"✅ HIT deleted")
        
    except ClientError as e:
        print(f"⚠️  Failed to delete HIT: {e}")
        print(f"   You may need to manually delete HIT {hit_id}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ MTurk Sandbox Test Complete!")
    print("=" * 70)
    print("\nYour MTurk sandbox setup is working correctly.")
    print("\nNext steps:")
    print("  1. Get your public URL (cloudflare tunnel / ngrok)")
    print("  2. Start your backend with --use-mturk --mturk-external-url=<your-url>")
    print("  3. Create HITs using the /api/mturk/create-hit endpoint")
    print("  4. Test with your MTurk sandbox worker account")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_mturk_sandbox()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# User Identity and Approval Tracking

This system tracks which users submit which actions and logs acceptance/rejection statistics for quality monitoring and worker performance analysis.

## Features

### 1. User Identity Collection (sim.html)

When a user first visits the sim.html interface, they are prompted to enter:
- **Name**: Their full name
- **Email**: Their email address

This information is:
- Stored in localStorage for the session
- Included with every action submission
- Used to track individual worker performance

### 2. Backend Tracking (state_manager.py)

The backend tracks:
- **Per-submission tracking**: Each action submission is linked to the user who submitted it
- **Pre-approval display**: When an action is shown for approval in monitor.html, the submitter's name/email is displayed
- **Execution history**: Each action in the execution_history includes `submitted_by` with user details

### 3. Monitor Display (monitor.html)

The pre-execution approval modal now shows:
- **Submitter information**: Name and email of users who submitted the action being reviewed
- Example: "📝 **Submitted by:** Alice Smith (alice@example.com), Bob Jones (bob@example.com)"

### 4. Approval Logging (dataset_manager.py)

Two types of logs are maintained:

#### a) Per-State Logs
After each state is completed, logs show:
```
State X: Accepted: [Alice, Bob], Rejected: [Charlie], 2/3 accepted (66.7%)
```

#### b) Episode Summary Logs
After each episode is saved, logs show:
- Per-user acceptance rates
- Overall episode acceptance rate

Example output:
```
Episode 5 User Approval Summary:
   Alice Smith (alice@example.com): 8/10 accepted (80.0%)
   Bob Jones (bob@example.com): 6/12 accepted (50.0%)
   Charlie Brown (charlie@example.com): 3/8 accepted (37.5%)
   Overall: 17/30 accepted (56.7%)
```

### 5. Log Files

All approval data is logged to:
- **Location**: `{dataset_root}/user_approval_log.jsonl`
- **Format**: JSON Lines (one JSON object per line)
- **Entry Types**:
  - `state_approval`: Per-state approval details
  - `episode_summary`: Per-episode user statistics

### 6. Analysis Script

Use the provided script to analyze logs:

```bash
python scripts/view_user_approval_stats.py outputs/lerobot_datasets/my_dataset
```

This displays:
- **Per-state approval details**: Who was accepted/rejected for each state
- **Episode summaries**: Per-user stats for each episode
- **Overall leaderboard**: Sorted by acceptance rate across all episodes
- **Overall acceptance rate**: System-wide statistics

Example output:
```
📈 OVERALL USER STATISTICS (Across All Episodes)
================================================================================

🏆 Leaderboard (by acceptance rate):
   🥇 #1: Alice Smith (alice@example.com)
        24/30 accepted (80.0%)
   🥈 #2: Bob Jones (bob@example.com)
        18/36 accepted (50.0%)
   🥉 #3: Charlie Brown (charlie@example.com)
        9/24 accepted (37.5%)

🌐 Overall Acceptance Rate: 51/90 (56.7%)
```

## Data Structure

### User Submission Tracking
```python
state_info["user_submissions"] = [
    {
        "name": "Alice Smith",
        "email": "alice@example.com",
        "action_index": 0  # Index in state_info["actions"]
    },
    ...
]
```

### Execution History with Users
```python
execution_history = [
    {
        "action": tensor([...]),
        "propensity": 0.33,
        "approval": 1,  # 1=approved, -1=rejected
        "submitted_by": [
            {"name": "Alice Smith", "email": "alice@example.com"}
        ]
    },
    ...
]
```

### Log Entry Format (state_approval)
```json
{
    "type": "state_approval",
    "episode_index": 5,
    "state_id": 42,
    "timestamp": 1704902400.0,
    "accepted_users": [
        {"name": "Alice Smith", "email": "alice@example.com"}
    ],
    "rejected_users": [
        {"name": "Bob Jones", "email": "bob@example.com"}
    ],
    "num_accepted": 1,
    "num_rejected": 1,
    "acceptance_rate": 0.5
}
```

### Log Entry Format (episode_summary)
```json
{
    "type": "episode_summary",
    "episode_index": 5,
    "timestamp": 1704902400.0,
    "user_stats": [
        {
            "name": "Alice Smith",
            "email": "alice@example.com",
            "accepted": 8,
            "rejected": 2,
            "total": 10,
            "acceptance_rate": 0.8
        }
    ],
    "overall_accepted": 17,
    "overall_rejected": 13,
    "overall_acceptance_rate": 0.567
}
```

## Use Cases

### 1. Worker Quality Monitoring
Identify workers with consistently low acceptance rates for:
- Additional training
- Performance feedback
- Quality control

### 2. Payment/Bonus Decisions
Use acceptance rates to:
- Determine bonus payments
- Identify top performers
- Make fair compensation decisions

### 3. Dataset Quality Analysis
Track which users' contributions are included in the final dataset:
- Understand data provenance
- Identify high-quality contributors
- Audit dataset composition

### 4. Training Data Attribution
For each datapoint in the dataset, you can trace:
- Who submitted the action
- Whether it was accepted or rejected
- How many alternatives were considered

## Privacy Considerations

- User identities are stored locally in localStorage (client-side)
- Backend logs include name and email for tracking purposes
- Logs are stored in the dataset directory
- Consider GDPR/privacy requirements for your use case

## Implementation Notes

### One-to-One Mapping Guarantee

The system maintains strict one-to-one mapping between:
- Actions submitted → Users who submitted them
- Actions approved/rejected → Users who submitted them

This is accomplished by:
1. Tracking `action_index` at submission time
2. Preserving this mapping through the approval pipeline
3. Storing `submitted_by` with each execution history entry

### Accuracy Critical

The user→action mapping is **critical** for fair compensation and quality control. The code includes:
- Explicit index tracking (`action_index`)
- Preservation through autofill clones
- Validation in execution history

Do not modify the mapping logic without careful testing.

## Testing

To test the system:

1. **Start the backend** with sim enabled
2. **Open sim.html** in multiple browser profiles (to simulate different users)
3. **Enter different identities** for each profile
4. **Submit actions** from each user
5. **Approve/reject** actions in monitor.html
6. **Check logs** with the analysis script

The logs should accurately reflect which users submitted which actions and which were approved/rejected.

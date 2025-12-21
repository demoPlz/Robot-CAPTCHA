# MTurk Integration Guide

## Overview

The crowdsourcing-ui system now supports Amazon Mechanical Turk (MTurk) integration for collecting human responses to critical robot states. This allows you to crowdsource responses at scale using MTurk's worker pool.

## Setup

### 1. Install Dependencies

```bash
pip install boto3
```

### 2. Configure AWS Credentials

Run `aws configure` and enter your MTurk requester credentials:

```bash
aws configure
```

You'll be prompted for:
- AWS Access Key ID
- AWS Secret Access Key  
- Default region name: `us-east-1`
- Default output format: (leave blank or `json`)

### 3. Test Sandbox Connection

Before going live, test your setup with the sandbox:

```bash
python backend/test_mturk_sandbox.py
```

This will:
- Verify your AWS credentials work
- Check your sandbox account balance
- Create and delete a test HIT
- Provide a preview URL to see how your HIT looks

### 4. Get Your Public URL

MTurk workers need to access your frontend. You have two options:

**Option A: Use your existing tunnel (Recommended - Auto-detected!)**

If you're already using `start_tunnel.sh` for deployment:
```bash
./start_tunnel.sh
```

The backend will **automatically detect** the tunnel URL from `/tmp/cloudflared.log`. No need to set `--mturk-external-url`!

**Option B: Manual tunnel setup**
```bash
# Cloudflare
cloudflared tunnel --url http://localhost:9000

# Or ngrok
ngrok http 9000
```

If using manual setup, copy the HTTPS URL and pass it via `--mturk-external-url`.

### 5. Start Backend with MTurk Enabled

**Easy mode (auto-detect tunnel):**
```bash
python backend/collect_data.py \
  --use-mturk \
  --mturk-sandbox \
  --mturk-reward=0.50 \
  [... other flags ...]
```

**Manual mode (specify URL):**
```bash
python backend/collect_data.py \
  --use-mturk \
  --mturk-sandbox \
  --mturk-external-url=https://your-public-url.com \
  --mturk-reward=0.50 \
  [... other flags ...]
```

**Important flags:**
- `--use-mturk`: Enable MTurk integration
- `--mturk-sandbox`: Use sandbox (test) environment
- `--mturk-production`: Use production environment (costs real money!)
- `--mturk-external-url`: (Optional) Your public URL - auto-detected if using `start_tunnel.sh`
- `--mturk-reward`: Payment per assignment in USD (default: $0.50)

## Usage

### Automatic HIT Creation

**HITs are created automatically** when critical states become ready for labeling.

A critical state is ready when:
1. Admin approves it in `monitor.html` (post-execution approval)
2. Prompt is set (text/video)
3. Simulation views are captured

When all three conditions are met, a HIT is automatically created with `N = required_responses_per_critical_state` assignments.

### Manual HIT Creation (Optional)

You can also manually create HITs via API:

```bash
curl -X POST http://localhost:5000/api/mturk/create-hit \
  -H "Content-Type: application/json" \
  -d '{"episode_id": 1, "state_id": 42}'
```

### Automatic HIT Lifecycle Management

The system automatically handles the complete HIT lifecycle:

1. **Creation**: HIT created when critical state is ready
2. **Collection**: Workers submit responses
3. **Expiration**: HIT expires when `max_assignments` reached
4. **Approval**: All assignments auto-approved within 30 seconds
5. **Deletion**: HIT deleted once all assignments approved

### Monitoring HITs

**Get status of specific HIT:**
```bash
curl "http://localhost:5000/api/mturk/hit-status?episode_id=1&state_id=42"
```

**Get all HITs:**
```bash
curl "http://localhost:5000/api/mturk/all-hits"
```

### Manual HIT Deletion (Rarely Needed)

```bash
curl -X POST http://localhost:5000/api/mturk/delete-hit \
  -H "Content-Type: application/json" \
  -d '{"episode_id": 1, "state_id": 42}'
```

## Worker Experience

1. Worker sees HIT on MTurk with your title/description
2. Clicks to accept, gets directed to your `sim_mturk.html` page
3. Sees the robot task interface embedded in an iframe
4. Completes the task (adjusts robot position, clicks Simulate, confirms)
5. After submission, sees "Task Completed" overlay
6. Clicks "Submit HIT" to finalize

## Architecture

### Components

1. **MTurkManager** (`backend/interface_managers/mturk_manager.py`)
   - Creates HITs with ExternalQuestion format
   - Tracks HIT status and assignment counts
   - Auto-approves all submissions in background thread
   - Handles HIT cleanup and deletion

2. **Flask Endpoints** (`backend/flask_app.py`)
   - `/api/mturk/create-hit`: Create HIT for a state
   - `/api/mturk/hit-status`: Get HIT status
   - `/api/mturk/all-hits`: Get all HITs
   - `/api/mturk/delete-hit`: Delete a HIT

3. **Frontend** (`src/pages/sim_mturk.html`)
   - Wrapper page for MTurk workers
   - Loads `sim.html` in iframe with `?mturk=1` parameter
   - Intercepts submission to prevent auto-refresh
   - Shows completion overlay instead

4. **Modified sim.html**
   - Detects MTurk mode via URL parameter
   - Posts message to parent window instead of redirecting
   - Disables auto-refresh behavior

### Data Flow

```
MTurk Worker
    ↓
sim_mturk.html (wrapper)
    ↓
sim.html?mturk=1 (iframe)
    ↓
/api/submit-goal (Flask)
    ↓
crowd_interface.record_response()
    ↓
mturk_manager.update_assignment_count()
    ↓
Auto-approval thread
    ↓
Worker gets paid
```

## Configuration Options

All options can be set in `crowd_interface_config.py` or via CLI flags:

```python
# MTurk settings
use_mturk: bool = False                           # Enable MTurk
mturk_sandbox: bool = True                        # Sandbox vs production
mturk_reward: float = 0.50                        # Payment in USD
mturk_assignment_duration_seconds: int = 600     # 10 minutes per task
mturk_lifetime_seconds: int = 3600               # 1 hour HIT lifetime
mturk_auto_approval_delay_seconds: int = 60      # Auto-approve after 1 min
mturk_title: str = "..."                         # HIT title
mturk_description: str = "..."                   # HIT description
mturk_keywords: str = "..."                      # Search keywords
mturk_external_url: str = None                   # Your public URL
```

## Best Practices

### Sandbox Testing

1. **Always test in sandbox first** - Sandbox uses fake money and fake workers
2. **Use your sandbox worker account** to test the full flow
3. **Verify the task works** before switching to production
4. **Check pricing** - Make sure your reward is appropriate

### Production Deployment

1. **Set competitive pricing** - Research similar HITs for fair wages
2. **Write clear instructions** - Workers need to understand the task
3. **Monitor quality** - Use your admin interface to review submissions
4. **Handle edge cases** - Workers may submit random responses
5. **Provide feedback** - Use RequesterFeedback when auto-approving

### Quality Control

MTurk integration auto-approves all submissions (fast payments = happy workers).

Quality control happens in **your admin interface** (`monitor.html`):
- Review submissions from critical states
- Accept or reject based on quality
- Your backend handles the real quality filtering
- MTurk payments are independent of your internal accept/reject

### Cost Optimization

- Set `mturk_lifetime_seconds` appropriately (don't leave HITs open too long)
- Use `required_responses_per_critical_state` wisely (more = higher cost)
- Monitor abandoned HITs and clean them up
- Consider qualification requirements to filter workers

## Switching to Production

When ready for real workers:

1. **Verify everything works in sandbox**
2. **Fund your MTurk requester account** (via AWS Management Console)
3. **Change one flag:**
   ```bash
   --mturk-production  # Instead of --mturk-sandbox
   ```
4. **Monitor closely** - Real money is being spent
5. **Start small** - Test with a few HITs before scaling

## Troubleshooting

### "Failed to initialize MTurk client"
- Check AWS credentials with `aws configure`
- Verify region is set to `us-east-1`
- Test with `python backend/test_mturk_sandbox.py`

### "Cannot create HIT: mturk_external_url not configured"
- Set `--mturk-external-url` flag when starting backend
- URL must be publicly accessible (use ngrok/cloudflare)
- URL must use HTTPS

### "Workers can't see my task"
- Verify your public URL is accessible
- Check that Flask server is running on correct port
- Test URL in incognito browser window
- Check CORS headers are set correctly

### "HITs not appearing on MTurk"
- Check you're looking at the right environment (sandbox vs production)
- Sandbox preview: `https://workersandbox.mturk.com/mturk/preview?groupId=<HIT_TYPE_ID>`
- Production preview: `https://worker.mturk.com/mturk/preview?groupId=<HIT_TYPE_ID>`

## Support

- AWS MTurk Documentation: https://docs.aws.amazon.com/mturk/
- MTurk Requester Help: https://requester.mturk.com/help
- Boto3 MTurk API: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk.html

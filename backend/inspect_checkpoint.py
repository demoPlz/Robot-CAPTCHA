import json, sys
try:
    with open('/home/yilong/.cache/huggingface/lerobot/switches/switches_c20_phase1_dcp/phase1_checkpoint.json', 'r') as f:
        data = json.load(f)
    print("Episodes in completed:", list(data['completed_states_by_episode'].keys()))
    states = data['completed_states_by_episode'].get('1', {})
    if '1284' in states:
        print("Keys in 1284:", list(states['1284'].keys()))
    else:
        print("1284 not found in completed states. Available states:", sorted([int(k) for k in states.keys()]))
except Exception as e:
    print("Error:", e)

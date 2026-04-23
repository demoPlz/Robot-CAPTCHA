import json, sys
data = json.load(open(sys.argv[1]))
val = list(data['states'].values())[0] if 'states' in data else list(data.values())[0]
print(json.dumps(val, indent=2)[:1000])

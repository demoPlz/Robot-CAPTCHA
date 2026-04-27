import glob
import re

replacement = """async function getCurrentCloudflaredUrl() {
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const portParam = urlParams.get('port');
    const h = (location.hostname === 'localhost' || location.hostname === '127.0.0.1') ? '127.0.0.1' : location.hostname;
    if (portParam) {
      console.log('[DEV] Using URL parameter port:', portParam);
      return `http://${h}:${portParam}`;
    }

    try {
      const response = await fetch('/backend-config.json');
      const data = await response.json();
      let backendUrl = data.backendUrl;
      // If config says localhost but client is on Tailscale/LAN, map localhost to client's requested LAN IP
      if (backendUrl && (backendUrl.includes('127.0.0.1') || backendUrl.includes('localhost'))) {
        if (location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
           const port = data.port || 9000;
           backendUrl = `http://${location.hostname}:${port}`;
        }
      }
      console.log('[CONFIG] Using backend URL:', backendUrl);
      return backendUrl || `http://${h}:9000`;
    } catch (e) {
      console.warn('[DEV] Could not read backend-config.json, using fallback');
      return `http://${h}:9000`;
    }
  } catch (error) {
    const h = (location.hostname === 'localhost' || location.hostname === '127.0.0.1') ? '127.0.0.1' : location.hostname;
    return `http://${h}:9000`;
  }
}"""

# Regex to match the old function and its contents
pattern = re.compile(
    r'(?m)^[ \t]*async function getCurrentCloudflaredUrl\(\) \{[\s\S]*?^[ \t]*\}(?=\s*\n[ \t]*(?:// Detect if page is served|let API_BASE_URL))'
)

files = glob.glob("/home/yilong/crowdsourcing-ui/src/pages/*.html")

for path in files:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find matches
    matches = pattern.findall(content)
    if matches:
        def replace_func(m):
            # preserve original indent of the function declaration
            indent = m.group(0).split('async')[0]
            indented_repl = '\n'.join([indent + line if line else line for line in replacement.split('\n')])
            return indented_repl

        new_content = pattern.sub(replace_func, content)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {path}")
    else:
        print(f"No match found in {path}")

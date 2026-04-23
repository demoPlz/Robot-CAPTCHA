import json
import base64
from pathlib import Path
from backend.interface_managers.dataset_manager import DatasetManager
import cv2

d = json.load(open("/home/yilong/.cache/huggingface/lerobot/sorting/sorting_c40_phase1_dcp/phase1_checkpoint.json"))
si = None
spool = d.get('async_state_pool', {})
cpool = d.get('completed_states_by_episode', {})
if "6" in cpool and "5558" in cpool["6"]:
    si = cpool["6"]["5558"]
elif ("6", "5558") in spool:
    si = spool[("6", "5558")]

if not si:
    # try any state
    for k, v in list(cpool.items())[:1]:
        print(f"Checking ep {k}")
        for st, s in list(v.items())[:1]:
            print(f"Using ep {k} st {st}")
            si = s
            break

obs_path = si.get('obs_path')
print(f"Found obs_path: {obs_path}")

dummy_dm = DatasetManager(Path("/home/yilong/.cache/huggingface/lerobot"))
obs = dummy_dm.load_obs_from_disk(obs_path)

if obs:
    print(list(obs.keys())[:10])
    cam_name = "cam_main"
    import numpy as np

    candidates = [
        f"observation.images.{cam_name}",
        f"observation.{cam_name}",
        cam_name,
        f"observation.images.{cam_name.replace('cam_', '')}"
    ]
    img_arr = None
    for k in candidates:
        if k in obs:
            val = obs[k]
            # duplicate _to_uint8_rgb logic
            if isinstance(val, (np.ndarray)):
                img_arr = val
            else:
                img_arr = val.numpy()
            
            if img_arr.dtype == np.float32 and img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).astype(np.uint8)
            
            # handle channel first
            if img_arr.ndim == 3 and img_arr.shape[0] in [1, 3]:
                img_arr = np.transpose(img_arr, (1, 2, 0))
            if img_arr.shape[-1] == 1:
                img_arr = np.repeat(img_arr, 3, axis=-1)
            break
            
    if img_arr is not None:
        print(f"Got img_arr: {img_arr.shape} dtype {img_arr.dtype}")
    else:
        print("Camera matrix not in chunk")

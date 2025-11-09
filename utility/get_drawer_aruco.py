import cv2
from PIL import Image

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

for marker_id in [17, 42]:
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)
    filename = f"aruco_{marker_id}_2in.png"
    cv2.imwrite(filename, img)

    # Reload with Pillow and add DPI metadata
    pil_img = Image.open(filename)
    pil_img.save(filename, dpi=(200, 200))

    print(f"Saved {filename} with 200 DPI (prints as 2x2 inches)")

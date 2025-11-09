# make_charuco_pdf.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Board parameters (edit if you change design) ----------
COLS, ROWS       = 5, 7                 # chessboard squares across, down
SQUARE_MM        = 35.0                 # edge of a chess square (mm)
MARKER_MM        = 26.0                 # ArUco marker edge (mm)
DICT_NAME        = cv2.aruco.DICT_4X4_50

# ---------- Page setup ----------
USE_US_LETTER    = True                 # set False for A4
DPI              = 600                  # bitmap DPI inside the PDF (high)
MARGIN_MM        = 10                   # white border around the board

if USE_US_LETTER:
    PAGE_W_MM, PAGE_H_MM = 215.9, 279.4   # 8.5" x 11"
else:
    PAGE_W_MM, PAGE_H_MM = 210.0, 297.0   # A4

# ---------- Derived dimensions ----------
board_w_mm = COLS * SQUARE_MM
board_h_mm = ROWS * SQUARE_MM

# Check it fits
assert board_w_mm + 2*MARGIN_MM <= PAGE_W_MM + 1e-6, "Board too wide for page."
assert board_h_mm + 2*MARGIN_MM <= PAGE_H_MM + 1e-6, "Board too tall for page."

# Pixel sizes for the bitmap we’ll place on the PDF
board_w_px = int(round(board_w_mm / 25.4 * DPI))
board_h_px = int(round(board_h_mm / 25.4 * DPI))
margin_px  = int(round(MARGIN_MM  / 25.4 * DPI))

# Make the Charuco board
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
try:
    board = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_MM/1000.0, MARKER_MM/1000.0, aruco_dict)
except TypeError:
    # Older OpenCV uses different constructor signature (same semantics)
    board = cv2.aruco.CharucoBoard_create(COLS, ROWS, SQUARE_MM/1000.0, MARKER_MM/1000.0, aruco_dict)

# Render board bitmap (high-res)
try:
    img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)
except AttributeError:
    img = board.draw((board_w_px, board_h_px), marginSize=0, borderBits=1)

# Place onto a white page canvas, centered
page_w_px = int(round(PAGE_W_MM / 25.4 * DPI))
page_h_px = int(round(PAGE_H_MM / 25.4 * DPI))
canvas = 255 * np.ones((page_h_px, page_w_px), dtype=np.uint8)

off_x = (page_w_px - board_w_px) // 2
off_y = (page_h_px - board_h_px) // 2
canvas[off_y:off_y+board_h_px, off_x:off_x+board_w_px] = img

# Mark the board origin (top-left corner of board, at off_x, off_y)
# OpenCV ChArUco origin is at the TOP-LEFT corner of the board
# Draw a small circle and crosshair at the origin
origin_x = off_x
origin_y = off_y
marker_size_px = int(round(5.0 / 25.4 * DPI))  # 5mm marker

# Convert to RGB for colored marker
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# Draw red circle at origin
cv2.circle(canvas_rgb, (origin_x, origin_y), marker_size_px, (0, 0, 255), -1)
# Draw white crosshair for visibility
cv2.line(canvas_rgb, (origin_x - marker_size_px*2, origin_y), 
         (origin_x + marker_size_px*2, origin_y), (255, 255, 255), 2)
cv2.line(canvas_rgb, (origin_x, origin_y - marker_size_px*2), 
         (origin_x, origin_y + marker_size_px*2), (255, 255, 255), 2)

# Add text label
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = 2
text_offset_x = origin_x + marker_size_px * 3
text_offset_y = origin_y + marker_size_px
cv2.putText(canvas_rgb, "ORIGIN (0,0,0)", (text_offset_x, text_offset_y), 
            font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

# Add coordinate system arrows
arrow_len_px = int(round(20.0 / 25.4 * DPI))  # 20mm arrows
# +X arrow (right, red)
cv2.arrowedLine(canvas_rgb, (origin_x, origin_y), 
                (origin_x + arrow_len_px, origin_y), (0, 0, 255), 3, tipLength=0.3)
cv2.putText(canvas_rgb, "+X", (origin_x + arrow_len_px + 10, origin_y), 
            font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
# +Y arrow (down, green)
cv2.arrowedLine(canvas_rgb, (origin_x, origin_y), 
                (origin_x, origin_y + arrow_len_px), (0, 255, 0), 3, tipLength=0.3)
cv2.putText(canvas_rgb, "+Y", (origin_x, origin_y + arrow_len_px + 25), 
            font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
# +Z arrow (out of page, blue - just show label)
cv2.putText(canvas_rgb, "+Z (out of page)", (origin_x + 10, origin_y - marker_size_px*2 - 10), 
            font, font_scale*0.7, (255, 0, 0), thickness, cv2.LINE_AA)

# Save as PDF at the correct physical size
fig_w_in = PAGE_W_MM / 25.4
fig_h_in = PAGE_H_MM / 25.4
fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
ax = plt.axes([0,0,1,1])
ax.imshow(canvas_rgb, vmin=0, vmax=255)
ax.axis('off')
plt.savefig("charuco_5x7_35mm_USletter.pdf", format="pdf", dpi=DPI)
plt.close(fig)

print("Wrote charuco_5x7_35mm_USletter.pdf")
print(f"Physical board area: {board_w_mm:.1f} mm x {board_h_mm:.1f} mm")
print("Origin marked with RED circle at top-left corner of board")
print("  +X = right (red arrow)")
print("  +Y = down (green arrow)")
print("  +Z = out of page (OpenCV default)")


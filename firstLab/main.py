import math
import functions

import numpy as np
from PIL import Image

# img_mat = np.zeros((800, 600, 3), dtype=np.uint8)
img_mat = np.full((1000, 1000, 3), (255, 255, 255), dtype=np.uint8)
for i in range(1, 14):
    x0, y0 = 500, 500
    angel = (i * 2 * math.pi) / 13
    x1, y1 = math.cos(angel) * 500 + 500, math.sin(angel) * 500 + 500
    functions.x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason(img_mat, x0, y0, x1, y1, (0, 0, 0))

img = Image.fromarray(img_mat, mode='RGB')
img.save('x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason.png')




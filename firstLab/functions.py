import math

import numpy as np

def dotted_line(image, x0, y0, x1, y1, count, color):
    step = 1 / count
    for t in np.arange(0, 1, step):
        x = round((1 - t) * x0 + t * x1)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color

def dotted_line_v2(image, x0, y0, x1, y1, color):
    count = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    step = 1 / count
    for t in np.arange(0, 1, step):
        x = round((1 - t) * x0 + t * x1)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line(image, x0, y0, x1, y1, color):
    x0 = int(x0)
    x1 = int(x1)

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line_hotfix_1(image, x0, y0, x1, y1, color):
    x0 = int(x0)
    x1 = int(x1)

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line_hotfix_2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    x0 = int(x0)
    x1 = int(x1)

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color


def x_loop_line_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    x0 = int(x0)
    x1 = int(x1)

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color


def x_loop_line_v2_no_y_calc(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = int(y0)
    dy = abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    x0 = int(x0)
    x1 = int(x1)

    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update


def x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = int(y0)
    dy =  2 * (x1 - x0) * abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    x0 = int(x0)
    x1 = int(x1)

    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 2 * (x1 - x0) * 0.5):
            derror -= 2 * (x1 - x0) * 1.0
            y += y_update

def bresenham_line(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = int(y0)
    dy =  2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    x0 = int(x0)
    x1 = int(x1)

    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

# img = ImageOps.flip(img) после понадобится

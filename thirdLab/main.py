import math
from math import floor, ceil, cos, sin

import numpy as np
from PIL import Image
from PIL import ImageOps



alpha = 0
beta = 90 * math.pi / 180
gamma = 0
Rx = np.array([[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]])
Ry = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
Rz = np.array([[cos(gamma), sin(gamma), 0], [-sin(gamma), cos(gamma), 0], [0, 0, 1]])
R = np.dot(Rx, np.dot(Ry, Rz))


def rotate_and_shift(vertex):
    new_coord = np.dot(R, np.array([vertex[0], vertex[1], vertex[2]]))
    new_coord = np.add(new_coord, np.array([0, -0.035, 1]))
    vertex[0] = new_coord[0]
    vertex[1] = new_coord[1]
    vertex[2] = new_coord[2]


def parse_obj(file_path):
        vertices = []
        faces = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    vertex = list(map(float, line.strip().split()[1:]))
                    rotate_and_shift(vertex)
                    vertices.append(vertex)
                elif line.startswith('f '):
                    face = [int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]]
                    faces.append(face)

        return vertices, faces

vertices, edges = parse_obj('C:\\Users\\DNS\\Downloads\\Telegram Desktop\\model_1.obj')


def calculate_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    first = np.array([x1 - x2, y1 - y2, z1 - z2])
    second = np.array([x1 - x0, y1 - y0, z1 - z0])

    cross_product = np.cross(first, second)
    return cross_product


def calculate_cos(normal):
    point_of_light = np.array([0, 0, 1])
    result = np.dot(normal, point_of_light)
    return result / (np.linalg.norm(normal) * np.linalg.norm(point_of_light))

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def getBarCoordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = float(((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = float(((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2))) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def printTriangle(image, x0nch, y0nch, z0, x1nch, y1nch, z1, x2nch, y2nch, z2, z_buffer, ax, ay):
    x0 = (ax * x0nch) / z0 + 1000
    x1 = (ax * x1nch) / z1 + 1000
    x2 = (ax * x2nch) / z2 + 1000

    y0 = (ay * y0nch) / z0 + 1000
    y1 = (ay * y1nch) / z1 + 1000
    y2 = (ay * y2nch) / z2 + 1000

    cos = calculate_cos(calculate_normal(x0nch, y0nch, z0, x1nch, y1nch, z1, x2nch, y2nch, z2))
    color = (-255 * cos, 0, 30)
    if cos < 0:
        xmin = floor(max(min(x0, x1, x2), 0))
        ymin = floor(max(min(y0, y1, y2), 0))

        xmax = ceil(min(max(x0, x1, x2), 2000))
        ymax = ceil(min(max(y0, y1, y2), 2000))

        for i in range(ymin, ymax, 1):
            for j in range(xmin, xmax, 1):
                first, second, third = getBarCoordinates(j, i, x0, y0, x1, y1, x2, y2)
                if first >= 0 and second >= 0 and third >= 0:
                    z_ = first * z0 + second * z1 + third * z2
                    if(z_ < z_buffer[i, j]):
                        z_buffer[i, j] = z_
                        image[i, j] = color


img_mat = np.full((2000, 2000, 3), (255, 255, 255), dtype=np.uint8)
z_buffer = np.full((2000, 2000), np.inf)

for poligon in edges:
    ax = 10000
    ay = 10000
    x0, y0, z0 = vertices[poligon[0]][0], vertices[poligon[0]][1], vertices[poligon[0]][2]
    x1, y1, z1 = vertices[poligon[1]][0], vertices[poligon[1]][1], vertices[poligon[1]][2]
    x2, y2, z2 = vertices[poligon[2]][0], vertices[poligon[2]][1], vertices[poligon[2]][2]
    printTriangle(img_mat,
                  x0, y0, z0,
                  x1, y1, z1,
                  x2, y2, z2,
                  z_buffer,
                  ax, ay
                  )


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('origin.png')

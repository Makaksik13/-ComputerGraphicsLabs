import math
from math import floor, ceil, cos, sin

import numpy as np
from PIL import Image
from PIL import ImageOps
from typing import List

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
    textures = []
    texture_coords = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            parts = line.split()
            if line.startswith('v '):
                vertex = list(map(float, parts[1:4]))
                rotate_and_shift(vertex)
                vertices.append(vertex)

            elif line.startswith('vt '):
                texture = list(map(float, parts[1:3]))
                texture_coords.append(texture)

            elif line.startswith('f '):
                face_vertices = []
                face_textures = []

                for part in parts[1:]:
                    vertex_data = part.split('/')

                    vertex_idx = int(vertex_data[0]) - 1
                    face_vertices.append(vertex_idx)

                    if len(vertex_data) > 1 and vertex_data[1]:
                        texture_idx = int(vertex_data[1]) - 1
                        face_textures.append(texture_idx)

                faces.append(face_vertices)
                if face_textures:
                    textures.append(face_textures)

    return vertices, faces, textures, texture_coords


def get_image_size(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
        return width, height


widthTexture, heightTexture = get_image_size('C:\\Users\\DNS\\PycharmProjects\\LabsForCG\\bunny-atlas.jpg')
vertices, edges, textures, textures_coords = parse_obj('C:\\Users\\DNS\\PycharmProjects\\LabsForCG\\model_1.obj')


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


def get_pixel_color(x, y):
    with Image.open('C:\\Users\\DNS\\PycharmProjects\\LabsForCG\\bunny-atlas.jpg') as img:
        rgb_img = img.convert('RGB')
        return rgb_img.getpixel((x, y))


def printTriangle(image, x0nch, y0nch, z0, x1nch, y1nch, z1, x2nch, y2nch, z2, z_buffer, ax, ay, I0, I1, I2, indexPlg, rgb_img):
    x0 = (ax * x0nch) / z0 + 1000
    x1 = (ax * x1nch) / z1 + 1000
    x2 = (ax * x2nch) / z2 + 1000

    y0 = (ay * y0nch) / z0 + 1000
    y1 = (ay * y1nch) / z1 + 1000
    y2 = (ay * y2nch) / z2 + 1000

    q1, q2 = textures_coords[textures[indexPlg][0]]
    q3, q4 = textures_coords[textures[indexPlg][1]]
    q5, q6 = textures_coords[textures[indexPlg][2]]

    cos = calculate_cos(calculate_normal(x0nch, y0nch, z0, x1nch, y1nch, z1, x2nch, y2nch, z2))
    #color = np.array([-255.0, 0, 0])
    if cos < 0:
        xmin = floor(max(min(x0, x1, x2), 0))
        ymin = floor(max(min(y0, y1, y2), 0))

        xmax = ceil(min(max(x0, x1, x2), 2000))
        ymax = ceil(min(max(y0, y1, y2), 2000))
        for i in range(ymin, ymax, 1):
            for j in range(xmin, xmax, 1):
                first, second, third = getBarCoordinates(j, i, x0, y0, x1, y1, x2, y2)
                coordTexture1 = widthTexture * (first * q1 + second * q3 + third * q5)
                coordTexture2 = heightTexture * (first * q2 + second * q4 + third * q6)
                I = first * I0 + second * I1 + third * I2
                if first >= 0 and second >= 0 and third >= 0:
                    z_ = first * z0 + second * z1 + third * z2
                    if(z_ < z_buffer[i, j]):
                        z_buffer[i, j] = z_
                        image[i, j] = np.array(rgb_img.getpixel((coordTexture1, coordTexture2))) * -I


img_mat = np.full((2000, 2000, 3), (255, 255, 255), dtype=np.uint8)
z_buffer = np.full((2000, 2000), np.inf)


def getArrayNormal():
    vertexToNeighborNormal = np.zeros((len(vertices), 3), dtype=np.float32)

    for plg in edges:
        x0, y0, z0 = vertices[plg[0]][0], vertices[plg[0]][1], vertices[plg[0]][2]
        x1, y1, z1 = vertices[plg[1]][0], vertices[plg[1]][1], vertices[plg[1]][2]
        x2, y2, z2 = vertices[plg[2]][0], vertices[plg[2]][1], vertices[plg[2]][2]

        normal = calculate_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        normal = normal / np.linalg.norm(normal)
        vertexToNeighborNormal[plg[0]] += normal
        vertexToNeighborNormal[plg[1]] += normal
        vertexToNeighborNormal[plg[2]] += normal

    for i in range(len(vertices)):
        norm = np.linalg.norm(vertexToNeighborNormal[i])
        vertexToNeighborNormal[i] = vertexToNeighborNormal[i] / norm

    return vertexToNeighborNormal


normals = getArrayNormal()
with Image.open('C:\\Users\\DNS\\PycharmProjects\\LabsForCG\\bunny-atlas.jpg') as img:
    rgb_img = ImageOps.flip(img.convert('RGB'))
    for index, polygon in enumerate(edges):
        ax = 10000
        ay = 10000
        x0, y0, z0 = vertices[polygon[0]][0], vertices[polygon[0]][1], vertices[polygon[0]][2]
        x1, y1, z1 = vertices[polygon[1]][0], vertices[polygon[1]][1], vertices[polygon[1]][2]
        x2, y2, z2 = vertices[polygon[2]][0], vertices[polygon[2]][1], vertices[polygon[2]][2]

        I0 = calculate_cos(normals[polygon[0]])
        I1 = calculate_cos(normals[polygon[1]])
        I2 = calculate_cos(normals[polygon[2]])

        printTriangle(img_mat,
                      x0, y0, z0,
                      x1, y1, z1,
                      x2, y2, z2,
                      z_buffer,
                      ax, ay,
                      I0, I1, I2,
                      index,
                      rgb_img
                      )


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('withTexture.png')

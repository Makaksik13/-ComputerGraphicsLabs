import math
import functions

import numpy as np
from PIL import Image
from PIL import ImageOps

def parse_obj(file_path):
        vertices = []
        faces = []

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    vertex = list(map(float, line.strip().split()[1:]))
                    for i in range(len(vertex) - 1):
                        vertex[i] = vertex[i] * 10000 + 1000
                    vertices.append(vertex)
                elif line.startswith('f '):
                    face = [int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]]
                    faces.append(face)

        return vertices, faces

img_mat = np.full((2000, 2000, 3), (255, 255, 255), dtype=np.uint8)
vertices, edges = parse_obj('C:\\Users\\DNS\\Downloads\\Telegram Desktop\\model_1.obj')

for edge in edges:
    prevVertex = edge[0]
    for i in range(1, len(edge)):
    #img_mat[round(vertices[i][1]) + 1000, round(vertices[i][0]) + 1000] = (0,0,0)
        functions.bresenham_line(img_mat, vertices[prevVertex][0], vertices[prevVertex][1], vertices[edge[i]][0], vertices[edge[i]][1], (0,0,0))
        prevVertex = edge[i];
    functions.bresenham_line(img_mat, vertices[prevVertex][0], vertices[prevVertex][1], vertices[edge[0]][0], vertices[edge[0]][1], (0,0,0))

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('objImage.png')





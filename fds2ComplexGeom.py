import os
import sys
from collections import defaultdict
import numpy as np
import json


class fds2ComplexGeom:
    def __init__(self, fds_input_location,tree_id):
        self.__treeID = tree_id
        self.__fds_input_location = fds_input_location
        self.__voxalSize = {}
        self.__meshBounds = {}
        self.__rowSpacing = 0
        self.__colSpacing = 0
        self.__treeList = []
        self.readInFDS_Mesh()
        self.readInTreeLocations()
        self.readInFDS_OBST()
        self.complex_geom()
        print()

    def readInFDS_OBST(self):
        self.topography = defaultdict(lambda: {})
        last_key = ""
        with open(self.__fds_input_location) as f:
            lines = f.readlines()

        lineCounter = 0
        while lineCounter < len(lines):
            current_line = lines[lineCounter]
            while "/" not in lines[lineCounter]:

                lineCounter += 1
                current_line = current_line + lines[lineCounter]
            if "&OBST" in current_line:

                XB = [
                    float(point)
                    for point in current_line.split("XB=")[1].split(",")[:6]
                ]
                self.topography[XB[0]][XB[2]] = XB[-1]
                last_key = XB[0]
                self.__colSpacing = abs(XB[2] - XB[3])

                self.__rowSpacing = abs(XB[0] - XB[1])

            lineCounter += 1

        self.nrows = len(self.topography[last_key])
        self.ncols = len(self.topography)

    def readInTreeLocations(self):
        with open(self.__fds_input_location) as f:
            lines = f.readlines()

        lineCounter = 0
        while lineCounter < len(lines):
            current_line = lines[lineCounter]
            while "/" not in lines[lineCounter]:
                lineCounter += 1
                current_line = current_line + lines[lineCounter]
                if "&INIT" in current_line and self.__treeID in current_line:
                    treeLine = current_line.replace("/", "").replace("\n", "")
                    XYZ_string = treeLine.split('XYZ=')[1].split(',')[:3]

                    x= float(XYZ_string[0])

                    y = float(XYZ_string[1])

                    z = float(XYZ_string[0])

                    tree_radius = treeLine.split('RADIUS=')[1].split(',')[0]


                    tree_height = treeLine.split('HEIGHT=')[1].split(',')[0]

                    currentTree = {"x":x,
                                   "y":y,
                                   "crownBaseHeight":z,
                                   "crownRadius":tree_radius,
                                   "crownHeight":tree_height,
                                   "height":z}
                    self.__treeList.append(currentTree)
            lineCounter += 1
    def readInFDS_Mesh(self):
        with open(self.__fds_input_location) as f:
            lines = f.readlines()

        lineCounter = 0
        while lineCounter < len(lines):
            current_line = lines[lineCounter]
            while "/" not in lines[lineCounter]:
                lineCounter += 1
                current_line = current_line + lines[lineCounter]
            if "&MESH" in current_line:
                mesh_line = current_line.replace("/", "").replace("\n", "")

                IJK = mesh_line.split("IJK=")[1].split(",")[:3]
                XB = mesh_line.split("XB=")[1].split(",")[:6]
                print(IJK, XB)
                self.__meshBounds = {
                    "X_MIN": float(XB[0]),
                    "Y_MIN": float(XB[2]),
                    "Z_MIN": float(XB[4]),
                    "X_MAX": float(XB[1]),
                    "Y_MAX": float(XB[3]),
                    "Z_MAX": float(XB[5]),
                    "I": float(IJK[0]),
                    "J": float(IJK[1]),
                    "K": float(IJK[2]),
                }
                self.__voxalSize = {
                    "X": (self.__meshBounds["X_MAX"] - self.__meshBounds["X_MIN"])
                    / self.__meshBounds["I"],
                    "Y": (self.__meshBounds["Y_MAX"] - self.__meshBounds["Y_MIN"])
                    / self.__meshBounds["J"],
                    "Z": (self.__meshBounds["Z_MAX"] - self.__meshBounds["Z_MIN"])
                    / self.__meshBounds["K"],
                }
            lineCounter += 1

    def complex_geom(self) -> str:
        """
        zMin: min elevation
        zMax: max elevation
        total_lat_distance_in_meters : distance of simulation horizontally
        total_long_distance_in_meters : distance of simulation area vertically
        right_index_value: right most index value of geotiff
        top_index_value: top most index value of geotiff
        left_index_value: left most index value of geotiff
        bottom_index_value: bottom most index value of geotiff

        Returns:
            str: complex gemity infomation for fds inputfile
        """

        vertices = np.zeros(((self.nrows + 1) * (self.ncols + 1) * 2, 3))

        vertCounter = 0

        offset = (self.nrows + 1) * (self.ncols + 1)
        colKeys = sorted(self.topography.keys())[::-1]
        for j in range(self.ncols):
            rowKeys = sorted(self.topography[colKeys[j]].keys())[::-1]
            for i in range(self.nrows):
                x1 = colKeys[j]
                x2 = colKeys[j] + self.__colSpacing
                z1 = rowKeys[i]
                z2 = rowKeys[i] + self.__rowSpacing
                y1 = self.__meshBounds["Z_MIN"]
                y2 = self.topography[colKeys[j]][rowKeys[i]]

                vertices[vertCounter] = [x2, z2, y2]
                vertices[vertCounter + offset] = [
                    x2,
                    z2,
                    y1,
                ]
                vertCounter += 1
                if j == self.ncols - 1:
                    vertices[vertCounter + self.nrows] = [
                        x2,
                        z1,
                        y2,
                    ]
                    vertices[vertCounter + self.nrows + offset] = [x2, z1, y1]

                if i == self.nrows - 1:
                    vertices[vertCounter] = [x1, z2, y2]
                    vertices[vertCounter + offset] = [x1, z2, y1]
                    vertCounter += 1
                    if j == self.ncols - 1:
                        vertices[vertCounter + self.nrows] = [x1, z1, y2]
                        vertices[vertCounter + self.nrows + offset] = [x1, z1, y1]

        vertCounter += self.nrows + 1

        faces = []
        faceCounter = 1
        """
        #        C---A
        #        | · |  
        #        D---B
        """

        # Top topography
        for j in range(self.ncols):
            for i in range(self.nrows):
                A = faceCounter
                B = faceCounter + 1
                C = faceCounter + self.nrows + 1
                D = faceCounter + self.nrows + 2

                # face 1  A -> B -> C
                faces.append([A, B, C])
                # face 2  B -> C -> D
                faces.append([C, B, D])
                faceCounter += 1
            faceCounter += 1

        # Right Side
        for i in range(1, self.nrows + 1):
            C = i
            A = i + offset
            D = i + 1
            B = i + offset + 1
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Left Side
        for i in range((offset - self.nrows), offset):
            A = i
            B = i + 1
            D = i + offset + 1
            C = i + offset
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Top Side
        for i in range(1, self.ncols + 1):
            A = (self.nrows + 1) * (i) + 1
            C = (self.nrows + 1) * (i - 1) + 1
            B = offset + (self.nrows + 1) * (i) + 1
            D = offset + (self.nrows + 1) * (i - 1) + 1
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Bottom Side
        for i in range(1, self.ncols + 1):
            A = (self.nrows + 1) * (i)
            B = offset + (self.nrows + 1) * (i)
            C = (self.nrows + 1) * (i + 1)
            D = (self.nrows + 1) * (i + 1) + offset
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Bottom Bottom
        faceCounter = 1
        for j in range(self.ncols):
            for i in range(self.nrows):
                C = offset + faceCounter
                A = offset + (self.nrows + 1) + faceCounter
                B = offset + (self.nrows + 1) + faceCounter + 1
                D = offset + faceCounter + 1

                # face 1  A -> B -> C
                faces.append([A, B, C])
                # face 2  B -> C -> D
                faces.append([C, B, D])
                # print([A, B, C])
                # face 2  B -> C -> D
                # print([C, B, D])
                faceCounter += 1
            faceCounter += 1

        for face_i in range(len(faces)):
            faces[face_i] = [faces[face_i][1], faces[face_i][0], faces[face_i][2]]
        self.jsonDict = {
            "meshData": self.__meshBounds,
            "verts": vertices,
            "faces": faces,
            "treeList": self.__treeList
        }

    def save2Json(self, filename):

        for key in self.jsonDict:
            print(key, type(self.jsonDict[key]))
            if type(self.jsonDict[key]) == np.ndarray:
                self.jsonDict[key] = self.jsonDict[key].tolist()

        with open(filename, "w") as f:
            json.dump(self.jsonDict, f)


def main(args):
    if len(args) == 0:
        app = fds2ComplexGeom("/home/trent/Trunk/Trunk/Trunk.fds","Generic Foliage")
        app.save2Json("data/testy.json")
        return
    if len(args) != 2:
        print("Useage : fds2ComplexGeom {fds Input File} {save Location}")
        return
    fds_loc = args[0]
    save_loc = args[1]
    app = fds2ComplexGeom(fds_loc)
    app.save2Json(save_loc)


if __name__ == "__main__":
    main(sys.argv[1:])

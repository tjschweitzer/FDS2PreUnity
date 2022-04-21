import sys
import json
import os
from collections import defaultdict

import h5py
import numpy as np


class fds2ComplexGeom:
    def __init__(self, fds_input_location, tree_id, non_terrain_obsts=[]):

        self.__treeID = tree_id
        self.__fds_input_location = fds_input_location
        self.__voxalSize = {}
        self.__meshBounds = {}
        self.__rowSpacing = 0
        self.__colSpacing = 0
        self.__treeList = []
        self.topography = defaultdict(lambda: {})
        self.__nonTerrainObsts = non_terrain_obsts
        self.read_in_fds_mesh()
        self.read_in_tree_locations()
        self.read_in_fds_obst()
        self.complex_geom()
        print()

    def read_in_fds_obst(self):
        counter = 0
        last_key = ""
        with open(self.__fds_input_location) as f:
            lines = f.readlines()

        line_counter = 0
        while line_counter < len(lines):
            current_line = lines[line_counter]
            while "/" not in lines[line_counter]:
                line_counter += 1
                current_line = current_line + lines[line_counter]
            current_line = current_line.lstrip()
            line_counter += 1

            if "&OBST" not in current_line[:5]:
                continue
            terrain_flag = [
                terrain not in current_line for terrain in self.__nonTerrainObsts
            ]

            if any(terrain_flag):
                print(current_line)
                counter += 1
                XB = [
                    float(point)
                    for point in current_line.split("XB=")[1].split(",")[:6]
                ]
                self.topography[XB[0]][XB[2]] = XB[-1]
                self.topography[XB[0]][XB[3]] = XB[-1]
                self.topography[XB[1]][XB[2]] = XB[-1]
                self.topography[XB[1]][XB[3]] = XB[-1]
                last_key = XB[0]
                self.__colSpacing = abs(XB[2] - XB[3])

                self.__rowSpacing = abs(XB[0] - XB[1])

        print(counter, "Terrain object found")
        self.n_rows = len(self.topography[last_key]) - 1
        self.n_cols = len(self.topography) - 1

    def read_in_tree_locations(self):
        with open(self.__fds_input_location) as f:
            lines = f.readlines()

        line_counter = 0
        while line_counter < len(lines):
            current_line = lines[line_counter]
            while "/" not in lines[line_counter]:
                line_counter += 1
                current_line = current_line + lines[line_counter]
            if "&INIT" in current_line and self.__treeID in current_line:
                tree_line = current_line.replace("/", "").replace("\n", "")
                xyz_string = tree_line.split("XYZ=")[1].split(",")[:3]

                x = float(xyz_string[0])

                y = float(xyz_string[1])

                z = float(xyz_string[2])

                tree_radius = tree_line.split("RADIUS=")[1].split(",")[0]

                tree_height = tree_line.split("HEIGHT=")[1].split(",")[0]

                current_tree = {
                    "x": x,
                    "y": y,
                    "crownBaseHeight": z,
                    "crownRadius": tree_radius,
                    "crownHeight": tree_height,
                    "height": z,
                }
                self.__treeList.append(current_tree)
            line_counter += 1

    def read_in_fds_mesh(self):
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

    def complex_geom(self) -> None:
        """
        Takes self.topography and builds a 3-D mesh


        Orientation of face variables
                C---A
                | · |
                D---B


        :var: vertices: Nx3 numpy array of all vertex points
        :var: vert_counter: keeps track of current vertex index
        :var: faces: array of all face indexes
        :var: face_counter: keeps track of current face index (one indexed)
        :var: col_keys: reverse sorted list of all index points of columns
        :var: row_keys: reverse sorted list of all index points of current row

        :rtype: None
        :return:  none
        """

        vertices = np.zeros(((self.n_rows + 1) * (self.n_cols + 1) * 2, 3))
        vert_counter = 0

        faces = []
        face_counter = 1

        offset = (self.n_rows + 1) * (self.n_cols + 1)
        col_keys = sorted(self.topography.keys())[::-1]

        for j in range(self.n_cols):

            row_keys = sorted(self.topography[col_keys[j]].keys())[::-1]

            for i in range(self.n_rows):

                x2 = col_keys[j]
                x1 = col_keys[j] - self.__colSpacing
                z2 = row_keys[i]
                z1 = row_keys[i] - self.__rowSpacing
                y1 = self.__meshBounds["Z_MIN"]
                y2 = self.topography[col_keys[j]][row_keys[i]]

                # top topography point
                vertices[vert_counter] = [x2, z2, y2]

                # bottom topography point
                vertices[vert_counter + offset] = [
                    x2,
                    z2,
                    y1,
                ]

                vert_counter += 1

                # checks if on the final column
                if j == self.n_cols - 1:
                    # top topography point final column
                    vertices[vert_counter + self.n_rows] = [
                        x1,
                        z2,
                        y2,
                    ]

                    # bottom topography point final column
                    vertices[vert_counter + self.n_rows + offset] = [x1, z2, y1]

                # checks if on the final row
                if i == self.n_rows - 1:

                    # top topography point final row
                    vertices[vert_counter] = [x2, z1, y2]

                    # bottom topography point final row
                    vertices[vert_counter + offset] = [x2, z1, y1]

                    vert_counter += 1

                    # checks if on the final row and column
                    if j == self.n_cols - 1:
                        # top topography point final column and row
                        vertices[vert_counter + self.n_rows] = [x1, z1, y2]

                        # bottom topography point final column anr row
                        vertices[vert_counter + self.n_rows + offset] = [x1, z1, y1]

        """
        #        C---A
        #        | · |  
        #        D---B
        """

        # Top topography
        for j in range(self.n_cols):
            for i in range(self.n_rows):
                A = face_counter
                B = face_counter + 1
                C = face_counter + self.n_rows + 1
                D = face_counter + self.n_rows + 2

                # face 1  A -> B -> C
                faces.append([A, B, C])
                # face 2  B -> C -> D
                faces.append([C, B, D])
                face_counter += 1
            face_counter += 1


        # Right Side
        for i in range(1, self.n_rows + 1):
            C = i
            A = i + offset
            D = i + 1
            B = i + offset + 1
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Left Side
        for i in range((offset - self.n_rows), offset):
            A = i
            B = i + 1
            D = i + offset + 1
            C = i + offset
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Top Side
        for i in range(1, self.n_cols + 1):
            A = (self.n_rows + 1) * (i) + 1
            C = (self.n_rows + 1) * (i - 1) + 1
            B = offset + (self.n_rows + 1) * (i) + 1
            D = offset + (self.n_rows + 1) * (i - 1) + 1
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Bottom Side
        for i in range(1, self.n_cols + 1):
            A = (self.n_rows + 1) * (i)
            B = offset + (self.n_rows + 1) * (i)
            C = (self.n_rows + 1) * (i + 1)
            D = (self.n_rows + 1) * (i + 1) + offset
            # face 1  A -> B -> C
            faces.append([A, B, C])
            # face 2   C-> B -> D
            faces.append([C, B, D])

        # Bottom Bottom
        face_counter = 1
        for j in range(self.n_cols):
            for i in range(self.n_rows):
                C = offset + face_counter
                A = offset + (self.n_rows + 1) + face_counter
                B = offset + (self.n_rows + 1) + face_counter + 1
                D = offset + face_counter + 1

                # face 1  A -> B -> C
                faces.append([A, B, C])
                # face 2  B -> C -> D
                faces.append([C, B, D])


                face_counter += 1
            face_counter += 1

        for face_i in range(len(faces)):
            faces[face_i] = [faces[face_i][1], faces[face_i][0], faces[face_i][2]]
        self.json_dict = {
            "meshData": self.__meshBounds,
            "verts": vertices,
            "faces": faces,
            "treeList": self.__treeList,
        }

    def save_to_json(self, filename):

        for key in self.json_dict:
            print(key, type(self.json_dict[key]))
            if type(self.json_dict[key]) == np.ndarray:
                self.json_dict[key] = self.json_dict[key].tolist()

        with open(filename, "w") as f:
            json.dump(self.json_dict, f)


    def write_h5py(self,  file_name):
        with h5py.File(f"{file_name}.hdf5", "w") as f:

            dict_group = f.create_group('meshData')
            for k, v in self.json_dict["meshData"].items():
                dict_group[k] = v
            f.create_dataset(
                "verts",
                data=np.array(self.json_dict["verts"], dtype=float),
            )
            f.create_dataset(
                "faces",
                data=np.array(self.json_dict["faces"], dtype=np.int64),
            )

            f.create_dataset(
                "tree_count",
                data=np.array([len(self.json_dict["treeList"])], dtype=np.int64),
            )
            for tree_count in range(len(self.json_dict["treeList"])):
                dict_group = f.create_group(f"tree_{tree_count}")
                for k, v in self.json_dict["treeList"][tree_count].items():
                    dict_group[k] = v
            print(f"{file_name}.hdf5", "saved")

        return self

def main(args):
    if len(args) == 0:

        app = fds2ComplexGeom(
            "/home/trent/Trunk/Trunk/Trunk.fds", "Generic Foliage", ["Trunk"]
        )
        # app.save_to_json("data\\testy.json")
        app.write_h5py("data/testy")

        return
    if len(args) < 2:
        print("Usage : fds2ComplexGeom {fds Input File} {save Location}")
        return
    fds_loc = args[0]
    save_loc = args[-1]
    if len(args) == 3:
        app = fds2ComplexGeom(fds_loc, args[1])
    else:
        app = fds2ComplexGeom(fds_loc, args[1], args[2:-1])

    app.save_to_json(save_loc)


if __name__ == "__main__":
    main(sys.argv[1:])

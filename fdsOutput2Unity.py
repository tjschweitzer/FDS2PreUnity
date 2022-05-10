import os.path
import sys
import time
import numpy as np
import glob
import json
import multiprocessing as mp
import fdsreader as fds


class fdsOutputToUnity:
    def __init__(
        self, fds_output_directory, fds_input_location, save_location, saveType="bin"
    ):
        """
        Converts plot3D data from fds into hdf5 data to be loaded into unity

        :param fds_output_directory:  Full path to where fds data was output
        :param fds_input_location:  Full path to fds input file
        :param save_location: Full path to where data should be saved one processed
        :param saveType:

        :raises StopIteration: if hrrpuv is not part of plot3D data dump
        """

        self.sim = fds.Simulation(fds_output_directory)
        pl_t1 = self.sim.data_3d[-1]
        try:
            self.hrr_idx = pl_t1.get_quantity_index("hrrpuv")
        except StopIteration:
            raise Exception("FDS Simulation requires HRRPUV in plot3d")
        mesh = self.sim.meshes[0]
        self.__timeList = np.array(self.sim.data_3d.times)
        for t in range(len(self.__timeList)):
            pl_t1 = self.sim.data_3d[t][mesh].file_path
            print(pl_t1)
        self.directory = fds_output_directory
        self.qFiles = glob.glob(fds_output_directory + "*.q")
        self.fileCounter = 0
        self.save_location = save_location
        self.allFilesDict = {}
        self.fds_input_location = fds_input_location

        # self.headerCountTitles = ['smoke', 'U-VELOCITY', 'V-VELOCITY', 'W-VELOCITY', 'fire']
        self.headerCountTitles = []
        self.read_in_fds()
        self.lenHeaderCountTitles = len(self.headerCountTitles)

        if len(self.headerCountTitles) == 0:
            print("No Dump line found in FDS input file ")
            return
        self.minValues = np.array([np.inf] * self.lenHeaderCountTitles)
        self.maxValues = np.zeros(self.lenHeaderCountTitles)
        self.save_function = (
            self.write_to_json if saveType == "json" else self.write2bin
        )

        self.filenames = self.group_files_by_time()
        self.my_mean = 0.0

    def read_in_fds(self):
        with open(self.fds_input_location) as f:
            lines = f.readlines()

        line_counter = 0
        while line_counter < len(lines):
            current_line = lines[line_counter]
            while "/" not in lines[line_counter]:
                line_counter += 1
                current_line = current_line + lines[line_counter]
            if "&DUMP" == current_line[:5]:
                dump_line = (
                    current_line.replace("/", "")
                    .replace("'", "")
                    .replace('"', "")
                    .replace("\n", "")
                )
                if "PLOT3D_QUANTITY" in dump_line:
                    self.headerCountTitles = []

                    plt3d_outputs = (
                        dump_line.split("PLOT3D_QUANTITY")[1].split("=")[1].split(",")
                    )
                    for plt3d_output in plt3d_outputs:
                        if "=" in plt3d_output:
                            break
                        self.headerCountTitles.append(plt3d_outputs)
                else:
                    self.headerCountTitles = [
                        "DENSITY",
                        "U-VELOCITY",
                        "V-VELOCITY",
                        "W-VELOCITY",
                        "HRRPUV",
                    ]

                break

            line_counter += 1

    @staticmethod
    def get_file_timestep(file) -> float:
        """
        Extracts the timestep from the filename.

        :param file: str: The file name to be processed.

        :rtype: float
        :return: time value of file
        """
        time_sec = int(file.split("_")[-2])
        time_milsec = int(file.split("_")[-1].split(".q")[0])
        return time_sec + time_milsec / 100.0

    def group_files_by_time(self):
        """
        Groups filenames by timestamp to allow for multi mesh simulations

        :return:     file_by_time: Dictionary <float,array>
                    KEY: Time
                   Value: Array of all files with same timestep

        """
        file_by_time = {}
        for file in self.qFiles:
            file_time = self.get_file_timestep(file)
            if file_time not in file_by_time.keys():
                file_by_time[file_time] = []
            file_by_time[file_time].append(file)
        return file_by_time

    def find_max_values_parallel(self):
        """
        Parallel function to find the max values for each time of data dumped
        :return: None
        """
        print(self.filenames)
        my_mean = []
        pool = mp.Pool()
        print("pool made")
        for j, returnValue in enumerate(
            pool.imap(self.get_values, self.filenames.keys())
        ):
            print(j, "Done")
            min_value = returnValue[1]
            max_value = returnValue[2]
            for i in range(self.lenHeaderCountTitles):
                self.maxValues[i] = (
                    self.maxValues[i]
                    if self.maxValues[i] > max_value[i]
                    else max_value[i]
                )
                self.minValues[i] = (
                    self.minValues[i]
                    if self.minValues[i] < min_value[i]
                    else min_value[i]
                )
            my_mean.append(returnValue[0][0])

        pool.close()
        pool.join()
        self.my_mean = np.mean(my_mean, axis=0)
        print("My Mean")
        print(self.my_mean)
        print("My Min")
        print(self.minValues)
        print("My Max")
        print(self.maxValues)

    def get_values(self, file_time):
        """
        Calculates minimum and max values for all types of data dumped

        :param file_time: time value of dump data to look at

        :return: mean values, minimum values, max values in an array
        :raises:array  3xN array: N is number of types of values dumped
        """
        print(file_time)
        self.fileCounter += 1
        min_value = [np.inf] * self.lenHeaderCountTitles

        max_value = [-np.inf] * self.lenHeaderCountTitles

        data_mean = []
        for file in self.filenames[file_time]:
            with open(file, "rb") as f:
                print(f"Opened file {file}")

                header = np.fromfile(f, dtype=np.int32, count=self.lenHeaderCountTitles)
                _ = np.fromfile(f, dtype=np.float32, count=7)
                (nx, ny, nz) = (header[1], header[2], header[3])

                data = np.fromfile(
                    f, dtype=np.float32, count=nx * ny * nz * self.lenHeaderCountTitles
                )
                data = np.reshape(
                    data,
                    (
                        data.shape[0] // self.lenHeaderCountTitles,
                        self.lenHeaderCountTitles,
                    ),
                    order="F",
                ).T

                data_min = np.min(data, axis=1)
                # data_min[0]=1.1
                data_min[4] = 15

                data_max = np.max(data, axis=1)
                file_percentile = []
                zero_counter = 0
                for data_i in data:
                    data_no_zeros = data_i[data_i > data_min[zero_counter]]
                    if len(data_no_zeros) < 1:
                        file_percentile.append(0.0)
                        continue
                    file_percentile.append(np.percentile(data_no_zeros, 99))
                    new_min = np.min(data_no_zeros)
                    if new_min < min_value[zero_counter]:
                        min_value[zero_counter] = new_min
                    if data_max[zero_counter] > max_value[zero_counter]:
                        max_value[zero_counter] = data_max[zero_counter]

                    zero_counter += 1
            data_mean.append(file_percentile)
        return data_mean, min_value, max_value

    def runParallel(self, multiMesh=None):
        """
        The function to read in and save data in parallel.

        :param multiMesh: default None: dictionary of all mesh infomation needed

        :return:
        """

        self.hrr_lower_limit = self.minValues[-1]
        self.smoke_lower_limit = self.my_mean[0]
        self.multi_mesh = multiMesh
        self.filenames = self.group_files_by_time()

        pool = mp.Pool()
        for i, temp_array in enumerate(
            pool.imap(self.q_file_to_dict, self.filenames.keys())
        ):
            print(i, temp_array)
        pool.close()
        pool.join()
        print(self.minValues)
        print(self.maxValues)

    @staticmethod
    def get_mesh_number(file_name):
        """
        Pulls mesh number from filename
        :param file_name:
        :return: Mesh Number
        """
        return int(file_name.split("_")[-3])

    def q_file_to_dict(self, file_time):

        self.fileCounter += 1
        empty_file = True
        current_fire_array = []
        current_density_array = []
        for file in self.filenames[file_time]:
            smoke_counter = 0
            with open(file, "rb") as f:

                counter = 0
                header = np.fromfile(f, dtype=np.int32, count=self.lenHeaderCountTitles)
                _ = np.fromfile(f, dtype=np.float32, count=7)
                (nx, ny, nz) = (header[1], header[2], header[3])

                data = np.fromfile(
                    f, dtype=np.float32, count=nx * ny * nz * self.lenHeaderCountTitles
                )
                data = np.reshape(
                    data,
                    (
                        data.shape[0] // self.lenHeaderCountTitles,
                        self.lenHeaderCountTitles,
                    ),
                    order="F",
                )

                mesh_number = self.get_mesh_number(file) - 1
                for i in range(nz):
                    for j in range(ny):
                        for k in range(nx):
                            if self.multi_mesh is not None:
                                mesh_row = mesh_number % (
                                    self.multi_mesh["I_UPPER"] + 1
                                )
                                mesh_col = np.floor(
                                    mesh_number / (self.multi_mesh["I_UPPER"] + 1)
                                )
                                mesh_height = np.floor(
                                    mesh_number
                                    / (
                                        (self.multi_mesh["I_UPPER"] + 1)
                                        * (self.multi_mesh["K_UPPER"] + 1)
                                    )
                                )

                                i += mesh_col * self.multi_mesh["K"]
                                j += mesh_height * self.multi_mesh["J"]
                                k += mesh_row * self.multi_mesh["I"]

                            if data[counter, 4] >= self.hrr_lower_limit:
                                empty_file = False
                                point_dict = {
                                    "X": int(i),
                                    "Y": int(j),
                                    "Z": int(k),
                                    "Datum": data[counter, 4],
                                }
                                smoke_counter += 1
                                current_fire_array.append(point_dict)

                            if round(data[counter, 0], 1) < 1.1:
                                empty_file = False
                                point_dict = {
                                    "X": int(i),
                                    "Y": int(j),
                                    "Z": int(k),
                                    "Datum": data[counter, 0],
                                }

                                current_density_array.append(point_dict)
                            counter += 1
        if not empty_file:
            # a = []
            # for i in current_density_array:
            #     a.append(round(i["Datum"],6))
            # a = Counter(a)
            # plt.bar(a.keys(), a.values())
            # plt.show()
            newfile = self.filenames[file_time][0]
            min_max_dict = {"min": list(self.minValues), "max": list(self.maxValues)}
            dictionary = {
                "fire": current_fire_array,
                "smoke": current_density_array,
                "configData": min_max_dict,
            }
            self.save_function(dictionary, newfile)

        return True

    @staticmethod
    def write_to_json(my_dict, file_name):
        new_file_name = (
            "_".join(file_name.split("_")[:-3])
            + "_1_"
            + "_".join(file_name.split("_")[-2:])
        )

        with open(f"{new_file_name.split('.q')[0]}.json", "w") as outfile:
            json.dump(my_dict, outfile)

    def write2bin(self, mydict, file_name):

        new_file_name = (
            "_".join(file_name.split("_")[:-3])
            + "_1_"
            + "_".join(file_name.split("_")[-2:])
        )
        new_file_name = new_file_name.split(".q")[0] + ".bin"
        new_file_name = self.save_location + os.path.basename(new_file_name)
        header_count_titles = [
            "smoke",
            "U-VELOCITY",
            "V-VELOCITY",
            "W-VELOCITY",
            "fire",
        ]

        header = np.array(
            [
                len(mydict[title]) if title in mydict else 0
                for title in header_count_titles
            ],
            dtype=np.int32,
        )
        print(file_name, header)
        print(f"Saved to {new_file_name}")
        with open(f"{new_file_name}", "wb") as outfile:

            np.ndarray.tofile(header, outfile)

            all_data = []
            for i in range(self.lenHeaderCountTitles):
                h = header_count_titles[i]

                if h not in mydict:
                    continue
                for j in range(header[i]):
                    point = mydict[h][j]
                    all_data.append(point["X"])
                    all_data.append(point["Y"])
                    all_data.append(point["Z"])

                    all_data.append(point["Datum"])

            print("min", self.minValues)
            print("max", self.maxValues)
            np.ndarray.tofile(np.array(self.minValues, dtype=np.float32), outfile)
            np.ndarray.tofile(np.array(self.maxValues, dtype=np.float32), outfile)
            np.ndarray.tofile(np.array(all_data, dtype=np.float32), outfile)
            print(file_name, "saved")

    def write_to_hdf5(self, mydict, file_name):

        new_file_name = (
            "_".join(file_name.split("_")[:-3])
            + "_1_"
            + "_".join(file_name.split("_")[-2:])
        )
        new_file_name = new_file_name.split(".q")[0] + ".bin"
        new_file_name = self.save_location + os.path.basename(new_file_name)
        header_count_titles = [
            "smoke",
            "U-VELOCITY",
            "V-VELOCITY",
            "W-VELOCITY",
            "fire",
        ]

        header = np.array(
            [
                len(mydict[title]) if title in mydict else 0
                for title in header_count_titles
            ],
            dtype=np.int32,
        )
        print(file_name, header)
        print(f"Saved to {new_file_name}")
        with open(f"{new_file_name}", "wb") as outfile:

            np.ndarray.tofile(header, outfile)

            all_data = []
            for i in range(self.lenHeaderCountTitles):
                h = header_count_titles[i]

                if h not in mydict:
                    continue
                for j in range(header[i]):
                    point = mydict[h][j]
                    all_data.append(point["X"])
                    all_data.append(point["Y"])
                    all_data.append(point["Z"])

                    all_data.append(point["Datum"])

            print("min", self.minValues)
            print("max", self.maxValues)
            np.ndarray.tofile(np.array(self.minValues, dtype=np.float32), outfile)
            np.ndarray.tofile(np.array(self.maxValues, dtype=np.float32), outfile)
            np.ndarray.tofile(np.array(all_data, dtype=np.float32), outfile)
            print(file_name, "saved")


# print( data, header[1:-1])


def main(args):

    start_time = time.time()
    if len(args) != 4:
        print(
            "Usage python fdsOutput2Unity.py {FDS Output Directory} "
            "{FDS Input File Path} {Output Directory} {Output FileType}"
        )

        fds_loc = "E:\\Trunk\\Trunk\\Trunk\\Trunk.fds"
        #
        fds_dir = "E:\\Trunk\\Trunk\\SableWindRE\\"
        save_location = "data"
        save_type = "bin"
    else:
        fds_loc = args[0]
        fds_dir = args[1]
        save_location = args[2]
        save_type = args[3]

    app = fdsOutputToUnity(
        fds_output_directory=fds_dir,
        fds_input_location=fds_loc,
        save_location=save_location,
        saveType=save_type,
    )

    app.find_max_values_parallel()
    app.runParallel()
    print(time.time() - start_time)


if __name__ == "__main__":
    main(sys.argv[1:])

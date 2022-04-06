# pylint: disable=line-too-long
import time
import os
from collections import defaultdict
import fdsreader as fds
from scipy.integrate import solve_ivp
import h5py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from matplotlib import cm

# %%


class FdsPathLines:
    def __init__(self, directory, fds_input_location):
        """

        :param directory: location of the fds output files
        :param fds_input_location: location of the fds input file


        :var self.sim:
        :var self.fds_input_location: fds_input_location
        :var self.t_span: t_span
        :var self.__directory: directory
        :var self.__qFiles: list of all plot 3d output files
        :var self.__timeList: list of all plot 3d time dumps
        :var self.__voxalSize: resolution of each voxal
        :var self.__maxVelocity: maximum velocity of any particle in the streamlines
        :var self.startingpoints: list of all starting points to be used in the ODE

        """

        self.sim = fds.Simulation(directory)
        self.fds_input_location = fds_input_location

        # self.__directory = directory
        # self.__qFiles = glob.glob(directory + "*.q")
        self.__time_list = np.array(self.sim.data_3d.times)
        self.__t_span = [np.min(self.__time_list), np.min(self.__time_list)]
        self.__voxel_size = {}
        self.__max_velocity = 0.0
        self.__max_re = 0.0
        self.__re_dict = defaultdict(lambda: [])
        self.starting_points = []
        self.__mesh_bounds = self.sim.meshes[0]
        self.__mesh_extent = self.sim.meshes[0].extent
        self.__n_bins = 800
        self.__ratio = 0.15
        self.__time_results = {}
        self.__min_voxel_count = 2.0
        self.filtered_time_results = {}

        self.distance_of_wind_streams_index = defaultdict(lambda: [])
        self.__set_voxel_size()

    def __set_voxel_size(self):
        """
        Calculates voxal size
        :return:
        """

        self.__voxel_size["vx"] = (
                                          self.__mesh_extent.x_end - self.__mesh_extent.x_start
        ) / (self.__mesh_bounds.dimension["x"] - 1)
        self.__voxel_size["vz"] = (
                                          self.__mesh_extent.z_end - self.__mesh_extent.z_start
        ) / (self.__mesh_bounds.dimension["z"] - 1)
        self.__voxel_size["vy"] = (
                                          self.__mesh_extent.y_end - self.__mesh_extent.y_start
        ) / (self.__mesh_bounds.dimension["y"] - 1)
        return self

    def get_position_from_index(self, x):
        """

        :param x:
        :return:
        """
        x_index = x[0]
        y_index = x[1]
        z_index = x[2]
        x_position = self.__mesh_extent.x_start + x_index * self.__voxel_size["vx"]
        y_position = self.__mesh_extent.y_start + y_index * self.__voxel_size["vy"]
        z_position = self.__mesh_extent.z_start + z_index * self.__voxel_size["vz"]
        return [x_position, y_position, z_position]

    def get_starting_points(self):
        """
        Creates a list of points  on the outer most voxals of OBSTS, one point per voxal
        :var x_min_value center point of minimum x voxal
        :var x_max_value center point of maximum x voxal
        :var y_min_value center point of minimum y voxal
        :var y_max_value center point of maximum y voxal

        :return:
        """
        x_min_value = self.__mesh_extent.x_start + self.__voxel_size["vx"] / 2.0
        x_max_value = self.__mesh_extent.x_end - self.__voxel_size["vx"] / 2.0
        y_min_value = self.__mesh_extent.y_start + self.__voxel_size["vy"] / 2.0
        y_max_value = self.__mesh_extent.y_end - self.__voxel_size["vy"] / 2.0
        with open(self.fds_input_location) as f:
            lines = f.readlines()

        line_counter = 0
        while line_counter < len(lines):
            current_line = lines[line_counter]
            if current_line == "\n":
                line_counter += 1
                continue
            while "/" not in lines[line_counter]:
                line_counter += 1
                current_line = current_line + lines[line_counter]

            line_counter += 1
            if "&OBST" not in current_line:
                continue
            mesh_line = current_line.replace("/", "").replace("\n", "")
            XB = [float(point) for point in mesh_line.split("XB=")[1].split(",")[:6]]

            if XB[0] <= x_min_value <= XB[1]:
                self.__add_ribbon_points(
                    [XB[0], XB[2], XB[5] + self.__voxel_size["vz"]],
                    [XB[0], XB[3], XB[5] + self.__voxel_size["vz"]],
                    int((XB[1] - XB[0]) / self.__voxel_size["vx"]),
                )

            if XB[0] <= x_max_value <= XB[1]:
                self.__add_ribbon_points(
                    [XB[1], XB[2], XB[5] + self.__voxel_size["vz"]],
                    [XB[1], XB[3], XB[5] + self.__voxel_size["vz"]],
                    int((XB[1] - XB[0]) / self.__voxel_size["vx"]),
                )
            if XB[2] <= y_min_value <= XB[3]:
                self.__add_ribbon_points(
                    [XB[0], XB[2], XB[5] + self.__voxel_size["vz"]],
                    [XB[1], XB[2], XB[5] + self.__voxel_size["vz"]],
                    int((XB[3] - XB[2]) / self.__voxel_size["vy"]),
                )
            if XB[2] <= y_max_value <= XB[3]:
                self.__add_ribbon_points(
                    [XB[0], XB[3], XB[5] + self.__voxel_size["vz"]],
                    [XB[1], XB[3], XB[5] + self.__voxel_size["vz"]],
                    int((XB[3] - XB[2]) / self.__voxel_size["vy"]),
                )

        return self

    def filter_streams_by_length(self):
        """
        This function removes all streamlines that total distance traveled is below a desired length.
        :return:
        """

        for _, time_data in enumerate(self.__time_results):
            data = self.__time_results[time_data]
            number_of_wind_streams = len(data)
            length_of_wind_streams = [len(x["y"][0]) for x in data]
            print(number_of_wind_streams)
            print(length_of_wind_streams)
            for i in range(number_of_wind_streams):
                distance_of_wind_stream = 0
                for j in range(1, len(data[i]["y"][0])):
                    point1 = np.array(
                        (
                            data[i]["y"][0][j - 1],
                            data[i]["y"][1][j - 1],
                            data[i]["y"][2][j - 1],
                        )
                    )
                    point2 = np.array(
                        (data[i]["y"][0][j], data[i]["y"][1][j], data[i]["y"][2][j])
                    )
                    p1_p2_distance = np.linalg.norm(point1 - point2)
                    distance_of_wind_stream += p1_p2_distance

                if (
                    distance_of_wind_stream
                    > np.min(list(self.__voxel_size.values())) * self.__min_voxel_count
                ):
                    self.distance_of_wind_streams_index[time_data].append(i)

    def __add_ribbon_points(self, starting_pont, ending_point, number_of_points):
        x_ = np.linspace(starting_pont[0], ending_point[0], number_of_points)
        y_ = np.linspace(starting_pont[1], ending_point[1], number_of_points)
        z_ = np.linspace(starting_pont[2], ending_point[2], number_of_points)
        points = np.stack((x_.flatten(), y_.flatten(), z_.flatten()), axis=1)

        self.starting_points.extend(points)
        return self

    def run_ode(self, time_step_index, reverse_integration=False):
        t_span = [
            min(self.__time_list[time_step_index:]),
            max(self.__time_list[time_step_index:]),
        ]
        current_results = []

        if reverse_integration:
            t_span[1] = 0.0

        for startCounter in range(len(self.starting_points)):
            y0 = self.starting_points[startCounter]

            result_solve_ivp = solve_ivp(
                self.get_velocity,
                t_span,
                y0,
                # rtol=1E-4, atol=1E-6,
            )
            result_with_velocity = self.__add_velocity_to_ode_data_frame(result_solve_ivp)
            current_result_max_vel = np.max(result_with_velocity["velocity"])
            if self.__max_velocity < current_result_max_vel:
                print(
                    f"new max Velocity {current_result_max_vel} changed from {self.__max_velocity}"
                )
                self.__max_velocity = current_result_max_vel

            result_with_re = self.add_reynolds_number(result_with_velocity)
            current_result_max_re = np.max(result_with_re["re"])
            self.__re_dict[self.__time_list[time_step_index]].append(
                current_result_max_re
            )
            if self.__max_re < current_result_max_re:
                print(f"new max RE {current_result_max_re} changed from {self.__max_re}")
                self.__max_re = current_result_max_re
            current_results.append(result_with_re)
        return current_results

    def StartODE(self, reverse_integration=True):

        for t_start in self.__time_list:
            if t_start > self.__t_span[1] or t_start < self.__t_span[0]:
                continue

            time_step_index = self.__get_closest_time_step_index(t_start)

            all_results = self.run_ode(time_step_index)
            if reverse_integration:
                backward = self.run_ode(time_step_index, True)
                all_results = self.combine_ode_frames(all_results, backward)
            self.__time_results[t_start] = all_results
        self.filter_streams_by_length()
        return self

    def combine_ode_frames(self, all_forward_data, all_backwards_data):
        return_values = all_backwards_data
        for i in range(len(all_backwards_data)):
            backwards_data = all_backwards_data[i]
            forward_data = all_forward_data[i]
            return_values[i]["t"] = backwards_data["t"][::-1]
            return_values[i]["y"][0] = backwards_data["y"][0][::-1]
            return_values[i]["y"][1] = backwards_data["y"][1][::-1]
            return_values[i]["y"][2] = backwards_data["y"][2][::-1]
            return_values[i]["velocity"] = backwards_data["velocity"][::-1] * -1
            return_values[i]["re"] = backwards_data["re"][::-1]

            return_values[i]["t"] = np.concatenate(
                (backwards_data["t"], forward_data["t"][1:])
            )

            y_ = [
                np.concatenate((backwards_data["y"][0], forward_data["y"][0][1:])),
                np.concatenate((backwards_data["y"][1], forward_data["y"][1][1:])),
                np.concatenate((backwards_data["y"][2], forward_data["y"][2][1:])),
            ]
            return_values[i]["y"] = y_
            return_values[i]["velocity"] = np.concatenate(
                (backwards_data["velocity"], forward_data["velocity"][1:])
            )

            return_values[i]["re"] = np.concatenate(
                (backwards_data["re"], forward_data["re"][1:])
            )

        return return_values



    def write_h5py(self, desired_directory, file_name_prefix):
        file_name = os.path.join(desired_directory, file_name_prefix)
        for re_counter, re_time in enumerate(self.__time_results):
            data = self.__time_results[re_time]
            number_of_wind_streams = len(data)
            length_of_wind_streams = [len(x["y"][0]) for x in data]
            time_string = f"{re_time}".split(".")[1]
            with h5py.File(f"{file_name}_{int(re_time)}_{time_string}.hdf5", "w") as f:
                f.create_dataset(
                    "maxValue",
                    data=[
                        np.max(np.array(self.__re_dict[re_time])),
                        np.max(np.array(self.__re_dict[re_time])),
                    ],
                )

                f.create_dataset(
                    "number_of_wind_streams",
                    data=np.array([number_of_wind_streams], dtype=np.int64),
                )
                f.create_dataset(
                    "length_of_wind_streams",
                    data=np.array(length_of_wind_streams, dtype=np.int64),
                )
                for i in range(number_of_wind_streams):
                    current_stream = []
                    for j in range(len(data[i]["y"][0])):
                        current_stream.append(
                            [
                                data[i]["t"][j],
                                data[i]["re"][j],
                                data[i]["y"][0][j],
                                data[i]["y"][1][j],
                                data[i]["y"][2][j],
                            ]
                        )
                        f.create_dataset(
                            f"windStream_{i+1}",
                            data=np.array(current_stream, dtype=float),
                        )

                print(f"{file_name}_{int(re_time)}_{time_string}.hdf5", "saved")

        return self

    def get_velocity(self, re_time, x):

        counter = self.__get_closest_time_step_index(re_time)
        plt_3d_data = self.sim.data_3d[counter]

        mesh = self.sim.meshes[0]
        # Select a quantity
        uvel_idx = plt_3d_data.get_quantity_index("U-VEL")
        vvel_idx = plt_3d_data.get_quantity_index("V-VEL")
        wvel_idx = plt_3d_data.get_quantity_index("W-VEL")
        index_values = self.get_index_values(x)
        u_vel_data = plt_3d_data[mesh].data[:, :, :, uvel_idx]
        u_velocity = u_vel_data[index_values[0], index_values[1], index_values[2]]
        v_vel_data = plt_3d_data[mesh].data[:, :, :, vvel_idx]
        v_velocity = v_vel_data[index_values[0], index_values[1], index_values[2]]
        w_vel_data = plt_3d_data[mesh].data[:, :, :, wvel_idx]
        w_velocity = w_vel_data[index_values[0], index_values[1], index_values[2]]

        return np.array([u_velocity, v_velocity, w_velocity])

    def get_index_values(self, x):
        x_index = (x[0] - self.__mesh_extent.x_start) / self.__voxel_size["vx"]
        if 0 > x_index or x_index > self.__mesh_bounds.dimension["x"]:
            return np.array([0, 0, 0], dtype=int)
        y_index = (x[1] - self.__mesh_extent.y_start) / self.__voxel_size["vy"]
        if 0 > y_index or y_index > self.__mesh_bounds.dimension["y"]:
            return np.array([0, 0, 0], dtype=int)
        z_index = (x[2] - self.__mesh_extent.z_start) / self.__voxel_size["vz"]
        if 0 > z_index or z_index > self.__mesh_bounds.dimension["z"]:
            return np.array([0, 0, 0], dtype=int)
        return np.array([x_index, y_index, z_index], dtype=int)

    def add_reynolds_number(self, data_set):
        """

        :param data_set:
        :return: same dataset with added velocity information
        """

        all_re = []
        all_times = data_set["t"]
        all_positions = data_set["y"]
        for i in range(len(all_positions[0])):
            current_position = np.array(
                [all_positions[0][i], all_positions[1][i], all_positions[2][i]]
            )
            current_time = all_times[i]
            all_re.append(self.__get_reynolds_number(current_position, current_time))
        data_set["re"] = np.array(all_re)
        return data_set

    @staticmethod
    def __add_velocity_to_ode_data_frame(data_set):
        """
        Adds velocity data set in to data frame        
        :rtype: object
        :param data_set: initial data frame
        :return: same dataset with added velocity information
        """

        all_speeds = [0.0]  # all particles start at 0 velocity

        all_times = data_set["t"]
        all_positions = data_set["y"]
        previous_position = np.array(
            [all_positions[0][0], all_positions[1][0], all_positions[2][0]]
        )
        for i in range(1, len(all_positions[0])):
            current_position = np.array(
                [all_positions[0][i], all_positions[1][i], all_positions[2][i]]
            )

            delta_time = all_times[i] - all_times[i - 1]
            squared_dist = np.sum((current_position - previous_position) ** 2, axis=0)
            dist = np.sqrt(squared_dist)
            if dist == 0.0 or delta_time == 0.0:
                speed = 0.0
            else:
                speed = dist / delta_time
            all_speeds.append(speed)
            previous_position = current_position

        data_set["velocity"] = np.array(all_speeds)
        return data_set

    def draw_stream_lines(self, animation=False):
        for _, wind_time in enumerate(self.distance_of_wind_streams_index):
            data = self.__time_results[wind_time]
            max_re = np.max(np.array(self.__re_dict[wind_time]))
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            for i in self.distance_of_wind_streams_index[wind_time]:
                print(data[i]["y"][0][:])
                wind_stream_x = data[i]["y"][0][:]
                wind_stream_y = data[i]["y"][1][:]
                wind_stream_z = data[i]["y"][2][:]
                wind_stream_re = data[i]["re"][:]

                # temp = np.max(wind_stream_re) / max_re
                ax.plot(wind_stream_x, wind_stream_y, wind_stream_z, c=cm.viridis(np.max(wind_stream_re) / max_re))
            if animation:
                for angle in range(0, 360 * 3):
                    ax.view_init(15, angle)
                    plt.draw()
                    plt.pause(0.001)
            else:
                plt.show()



    def __get_closest_time_step_index(self, re_time):
        closest_time_step_value = min(self.__time_list, key=lambda x: abs(x - re_time))
        closest_time_step_index = np.where(self.__time_list == closest_time_step_value)[
            0
        ][0]
        return int(closest_time_step_index)

    def __get_reynolds_matrix(self, re_time):
        time_step_index = self.__get_closest_time_step_index(re_time)
        plt_3d_data = self.sim.data_3d[time_step_index]

        mesh = self.sim.meshes[0]
        # Select a quantity
        try:
            dxeta_idx = plt_3d_data.get_quantity_index("dx/eta")
        except:
            print("dx/eta plot 3d data ot found ")
            return []

        re_data = plt_3d_data[mesh].data[:, :, :, dxeta_idx]

        return re_data

    def __get_reynolds_number(self, x, re_time):
        """
        takes the time and position and returns the nearest reynolds value
        :param x: Array Position [x,y,z]
        :param re_time: time value
        :return: reynolds value
        """
        re_data = self.__get_reynolds_matrix(re_time)
        if len(re_data) == 0:
            return None

        index_values = self.get_index_values(x)
        re_value = re_data[index_values[0]][index_values[1]][index_values[2]]

        return re_value

    def get_data_from_time(self, re_time):
        """

        :param re_time:
        :return:
        """
        if re_time in self.__time_results:
            return self.__time_results[re_time]
        return None

    def get_max_re(self):
        """
        returns max RE Value
        :return:
        """
        return self.__max_re

    def get_average_re_over_time(self, t_range):
        """
        Averages the reynolds number value over a range o times
        :param t_range: range o start and end time frame
        :return: ixjxk array
        """
        all_time_list = self.__time_list
        filtered_time_lists = None
        for t_start, t_end in t_range:
            filtered_time_list = all_time_list[all_time_list >= t_start]
            filtered_time_list = filtered_time_list[filtered_time_list <= t_end]
            if filtered_time_lists is None:
                filtered_time_lists = filtered_time_list
            else:
                filtered_time_lists = np.append(
                    filtered_time_lists, filtered_time_list, axis=0
                )

        re_average_matrix = self.__get_reynolds_matrix(filtered_time_list[0])
        for i in range(1, len(filtered_time_list)):
            re_average_matrix = re_average_matrix + self.__get_reynolds_matrix(
                filtered_time_list[i]
            )
        re_average_matrix = re_average_matrix / len(filtered_time_list)
        return np.array(re_average_matrix)

    def get_mean_std(self, re_matrix, re_range):
        """

        :param re_matrix:
        :return:
        """
        data_flatten = re_matrix.flatten()
        data_no_zero = data_flatten[data_flatten > 0.0]

        print(f"Min non Zero Value {np.min(data_no_zero)}")

        data_mean = np.mean(data_no_zero)
        data_std = np.std(data_no_zero)

        print(f"Standard Dev {data_std} Mean {data_mean}")
        if PLOT_FLAG:
            fig, ax = plt.subplots(figsize=(12, 6))

            # plt.title(f"{n_bins} Bins without Zero Values")
            plt.xlim([0, np.max(data_no_zero)])

            n_ranges, bins, patches = plt.hist(data_no_zero, bins=self.__n_bins)

            plt.axvline(
                data_mean, color="k", linestyle="dashed", linewidth=1, label="Mean"
            )
            plt.axvline(
                150, color="r", linestyle="dashed", linewidth=1, label="Turbulent Flow"
            )
            plt.axvline(
                40,
                color="g",
                linestyle="dashed",
                linewidth=1,
                label="Laminar Flow",
            )

            plt.xlabel("Reynolds Value")
            plt.ylabel("Voxel Count")

            ax.legend(prop={"size": 16})

            for patch_i,patch in enumerate(patches):
                patch.set_fc("grey")
                if bins[patch_i] >= 150:
                    patch.set_fc("red")
                if bins[patch_i] <= 40:
                    patch.set_fc("green")
            plt.show()
        else:

            n_ranges, bins = np.histogram(data_no_zero, bins=self.__n_bins)
        return_dict = {
            "n_ranges": n_ranges,
            "bins": bins,
            "mean": data_mean,
            "std": data_std,
            "sigmaOne": data_mean + data_std,
            "sigmaTwo": data_mean + 2.0 * data_std,
            "sigmaNegThree": data_mean - 3.0 * data_std,
            "sigmaNegTwo": data_mean - 2.0 * data_std,
            "sigmaNegOne": data_mean - data_std,
            "None": None,
            "zero": 0,
        }
        return return_dict

    def __plot_points_re_range(self, test_data, plot_range, plot_info):
        if isinstance(plot_range[0], str):
            plot_range[0] = plot_info[plot_range[0]]
        if isinstance(plot_range[1], str):
            plot_range[1] = plot_info[plot_range[1]]
        print(f"Points between {plot_range[0]} and {plot_range[1]}")
        x_1 = []
        y_1 = []
        z_1 = []
        c_1 = []
        for x_i in range(test_data.shape[0]):
            for y_i in range(test_data.shape[1]):
                for z_i in range(test_data.shape[2]):
                    if plot_range[0] < test_data[x_i, y_i, z_i]:
                        if (
                            plot_range[1] is None
                            or test_data[x_i, y_i, z_i] <= plot_range[1]
                        ):
                            x_1.append(x_i)
                            y_1.append(y_i)
                            z_1.append(z_i)
                            c_1.append(test_data[x_i, y_i, z_i])
        index_values = [x_1, y_1, z_1, c_1]
        position_values = np.array(
            [
                self.get_position_from_index(
                    [index_values[0][i], index_values[1][i], index_values[2][i]]
                )
                for i in range(len(index_values[0]))
            ]
        )
        if PLOT_FLAG:
            plt.figure(figsize=(10, 7))
            ax_1 = plt.axes(projection="3d")

            # Creating color map
            # my_cmap = plt.get_cmap("viridis")
            scatter_plot = ax_1.scatter3D(
                position_values.T[0],
                position_values.T[1],
                position_values.T[2],
                c=c_1,
                cmap=plt.get_cmap("viridis"),
            )
            ax_1.set_zlim(self.__mesh_extent.z_start, self.__mesh_extent.z_end)
            ax_1.set_ylim(self.__mesh_extent.y_start, self.__mesh_extent.y_end)
            ax_1.set_xlim(self.__mesh_extent.x_start, self.__mesh_extent.x_end)
            plt.colorbar(scatter_plot)
            plt.show()
        print(f"There are this many Points {len(x_1)}")

        return np.array(position_values), np.array(
            [index_values[3][i] for i in range(len(index_values[0]))]
        )

    def get_starting_positions(self, position_values, k_num, weighted):
        """

        :param position_values:
        :param k_num:
        :param weighted:
        :return:
        """
        position_df = pd.DataFrame(position_values)
        k_mean = KMeans(n_clusters=k_num)
        k_mean.fit(position_df, sample_weight=weighted)
        label = k_mean.predict(position_df)
        closest, _ = pairwise_distances_argmin_min(k_mean.cluster_centers_, position_df)
        print(f"Silhouette Score(n={k_num}): {silhouette_score(position_df, label)}")
        if PLOT_FLAG:
            fig = plt.figure(figsize=(8, 6))
            ax_1 = fig.add_subplot(1, 1, 1, projection="3d")
            ax_1.set_zlim(self.__mesh_extent.z_start, self.__mesh_extent.z_end)
            ax_1.set_ylim(self.__mesh_extent.y_start, self.__mesh_extent.y_end)
            ax_1.set_xlim(self.__mesh_extent.x_start, self.__mesh_extent.x_end)
            my_cmap = plt.get_cmap("viridis")
            ax_1.scatter(
                position_values.T[0],
                position_values.T[1],
                position_values.T[2],
                c=label,
                cmap=my_cmap,
            )

            plt.legend()
            plt.show()
        return np.array(
            [
                [
                    position_values.T[0][c],
                    position_values.T[1][c],
                    position_values.T[2][c],
                ]
                for c in closest
            ]
        )

    def get_all_starting_points(self, t_range, re_range, k_means):
        """

        :param t_range: range of time used to computing average
        :param re_range: range of reynolds numbers
        :param k_means: number of clusters used in k means
        :return: list of all starting points
        """
        average_data_over_range = self.get_average_re_over_time(t_range)
        average_data_over_range *= self.__ratio
        plot_info = self.get_mean_std(average_data_over_range, re_range)

        position_values, weight = self.__plot_points_re_range(
            average_data_over_range, re_range, plot_info
        )
        starting_positions = self.get_starting_positions(
            position_values, k_means, weighted=weight
        )

        return starting_positions


PLOT_FLAG = True

#%%
def main():
    fds_loc = "/home/trent/Trunk/Trunk/Trunk.fds"
    fds_dir = "/home/trent/Trunk/TimeDelay"
    # fds_loc = "/home/kl3pt0/Trunk/Trunk/Trunk.fds"
    # fds_dir = "/home/kl3pt0/Trunk/Fire"

    # fds_loc = "E:\Trunk\Trunk\Trunk\Trunk.fds"
    #
    # fds_dir = "E:\Trunk\Trunk\\SableWindRE\\"

    start_time = time.perf_counter()
    app = FdsPathLines(fds_dir, fds_loc)
    # app.EvaluateReynoldsValues()

    time_sets = [[0, 10]]
    re_ranges_and_k_means = [[0, 40, 24], [150, "None", 76]]

    all_starting_points = None
    for re_min, re_max, k_means in re_ranges_and_k_means:
        starting_postions = app.get_all_starting_points(
            time_sets, [re_min, re_max], k_means
        )

        if all_starting_points is None:
            all_starting_points = starting_postions
            continue
        all_starting_points = np.append(all_starting_points, starting_postions, axis=0)

    app.starting_points = all_starting_points

    # for i in range(3,21,2):
    #     app.startingPointsRibbon([0,0,i],[20,0,i],10)
    #     app.startingPointsRibbon([0,0,i],[0,20,i],10)
    # app.startingpoints = [[np.random.rand()*20, 0, np.random.rand()*17.5+2.5] for i in range(50)]
    # app.startingpoints = np.append(app.startingpoints,[[ 0,np.random.rand()*20, np.random.rand()*17.5+2.5] for i in range(50)],axis=0)

    # app.getStartingPoints()
    app.StartODE(reverse_integration=True)

    app.filter_streams_by_length()
    # app.drawPlot3D()
    app.write_h5py("data", "weightedMeans")
    print(time.process_time() - start_time)


if __name__ == "__main__":
    main()

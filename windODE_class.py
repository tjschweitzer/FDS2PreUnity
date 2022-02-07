# Import the required modules
import glob
import sys
import time
from collections import defaultdict
import fdsreader as fds
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class windODE:
    def __init__(
        self,
        directory,
        fds_input_location,
        t_span,
    ):
        """

        :param directory: location of the fds output files
        :param fds_input_location: location of the fds input file
        :param t_span: time frame for ODE to run over

        :vars self.sim: fdsreader object
        :vars self.fds_input_location: fds_input_location
        :vars self.t_span: t_span
        :vars self.__directory: directory
        :vars self.__qFiles: list of all plot 3d output files
        :vars self.__timeList: list of all plot 3d time dumps
        :vars self.__meshBounds: boundaries of entire mesh
        :vars self.__voxalSize: resolution of each voxal
        :vars self.__maxVelocity: maximum velocity of any particle in the streamlines
        :vars self.startpoints: list of all starting points to be used in the ODE

        """

        self.sim = fds.Simulation(directory)
        self.fds_input_location = fds_input_location
        self.t_span = t_span
        self.__directory = directory
        self.__qFiles = glob.glob(directory + "*.q")
        self.__timeList = np.array(self.sim.data_3d.times)
        self.__meshBounds = {}
        self.__voxalSize = {}
        self.__maxVelocity = 0.0
        self.startpoints = []

        self.getMeshBounds()



    def getMeshBounds(self):
        self.mesh = self.sim.meshes[0]
        self.mesh_extent = self.sim.meshes[0].extent

        self.__voxalSize["X"] = (self.mesh_extent.x_end - self.mesh_extent.x_start) / (
            self.mesh.dimension["x"] - 1
        )
        self.__voxalSize["Z"] = (self.mesh_extent.z_end - self.mesh_extent.z_start) / (
            self.mesh.dimension["z"] - 1
        )
        self.__voxalSize["Y"] = (self.mesh_extent.y_end - self.mesh_extent.y_start) / (
            self.mesh.dimension["y"] - 1
        )
        return self

    def getStartingPoints(self):

        X_Min_Value = (
                self.mesh_extent.x_start + self.__voxalSize["X"] / 2.0
        )  # Center point of mesh
        X_Max_Value = (
                self.mesh_extent.x_end - self.__voxalSize["X"] / 2.0
        )  # Center point of mesh
        Y_Min_Value = (
                self.mesh_extent.y_start + self.__voxalSize["Y"] / 2.0
        )  # Center point of mesh
        Y_Max_Value = (
                self.mesh_extent.y_end - self.__voxalSize["Y"] / 2.0
        )  # Center point of mesh

        with open(self.fds_input_location) as f:
            lines = f.readlines()

        lineCounter = 0
        while lineCounter < len(lines):
            current_line = lines[lineCounter]
            if current_line == "\n":
                lineCounter += 1
                continue
            while "/" not in lines[lineCounter]:
                lineCounter += 1
                current_line = current_line + lines[lineCounter]

            lineCounter += 1
            if "&OBST" not in current_line:
                continue
            mesh_line = current_line.replace("/", "").replace("\n", "")
            XB = [float(point) for point in mesh_line.split("XB=")[1].split(",")[:6]]

            if XB[0] <= X_Min_Value <= XB[1]:
                self.startingPointsRibbon(
                    [XB[0], XB[2], XB[5] + self.__voxalSize["Z"]],
                    [XB[0], XB[3], XB[5] + self.__voxalSize["Z"]],
                    int((XB[1] - XB[0]) / self.__voxalSize["X"]),
                )

            if XB[0] <= X_Max_Value <= XB[1]:
                self.startingPointsRibbon(
                    [XB[1], XB[2], XB[5] + self.__voxalSize["Z"]],
                    [XB[1], XB[3], XB[5] + self.__voxalSize["Z"]],
                    int((XB[1] - XB[0]) / self.__voxalSize["X"]),
                )
            if XB[2] <= Y_Min_Value <= XB[3]:
                self.startingPointsRibbon(
                    [XB[0], XB[2], XB[5] + self.__voxalSize["Z"]],
                    [XB[1], XB[2], XB[5] + self.__voxalSize["Z"]],
                    int((XB[3] - XB[2]) / self.__voxalSize["Y"]),
                )
            if XB[2] <= Y_Max_Value <= XB[3]:
                self.startingPointsRibbon(
                    [XB[0], XB[3], XB[5] + self.__voxalSize["Z"]],
                    [XB[1], XB[3], XB[5] + self.__voxalSize["Z"]],
                    int((XB[3] - XB[2]) / self.__voxalSize["Y"]),
                )

        return self

    def filterOutStreamsByLength(self):
        """
        This function removes all streamlines that total distance traveled is below a desired length.
        :return:
        """
        self.filteredTImeResults = {}

        self.distanceofWindStreams_index = defaultdict(lambda: [])
        allData = self.timeReasults
        for time in allData.keys():

            data = allData[time]
            numberofWindstreams = len(data)
            lengthofWindStreams = [len(x["y"][0]) for x in data]
            print(numberofWindstreams)
            print(lengthofWindStreams)
            for i in range(numberofWindstreams):
                distanceofWindStream = 0
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
                    distanceofWindStream += p1_p2_distance

                if distanceofWindStream > np.min(list(self.__voxalSize.values())) * 2.0:
                    self.distanceofWindStreams_index[time].append(i)
        print()

    def startingPointsRibbon(self, starting_pont, ending_point, number_of_points):
        x_ = np.linspace(starting_pont[0], ending_point[0], number_of_points)
        y_ = np.linspace(starting_pont[1], ending_point[1], number_of_points)
        z_ = np.linspace(starting_pont[2], ending_point[2], number_of_points)
        points = np.stack((x_.flatten(), y_.flatten(), z_.flatten()), axis=1)

        self.startpoints.extend(points)
        return self

    def runODE(self, timedependent=True):
        self.timeReasults = {}
        for t_start in self.__timeList:
            if t_start > self.t_span[1] or t_start < self.t_span[0]:
                break

            closest_timeStep = min(self.__timeList, key=lambda x: abs(x - t_start))
            counter = np.where(self.__timeList == closest_timeStep)[0][0]
            all_results = []
            for startCounter in range(len(self.startpoints)):

                y0 = self.startpoints[startCounter]
                t_span = [
                    min(self.__timeList[counter:]),
                    max(self.__timeList[counter:]),
                ]
                result_solve_ivp = solve_ivp(
                    self.get_velocity, t_span, y0, args=[t_span[0], timedependent]
                )
                result_with_velocity = self.addVelocity(result_solve_ivp)
                current_result_max_vel = np.max(result_with_velocity["velocity"])
                if self.__maxVelocity < current_result_max_vel:
                    print(
                        f"new max Velocity {current_result_max_vel} changed from {self.__maxVelocity}"
                    )
                    self.__maxVelocity = current_result_max_vel
                all_results.append(result_with_velocity)
            self.timeReasults[t_start] = all_results
        return self

    def write2bin(self, fileName):
        allData = self.timeReasults
        maxVel = self.__maxVelocity
        print(f"Max {maxVel}")
        for time in allData.keys():
            data = allData[time]
            numberofWindstreams = len(data)
            lengthofWindStreams = [len(x["y"][0]) for x in data]
            print(numberofWindstreams)
            print(lengthofWindStreams)
            time_string = f"{time}".split(".")[1]
            with open(f"{fileName}_{int(time)}_{time_string}.binwind", "wb") as outfile:

                np.ndarray.tofile(np.array([maxVel], dtype=np.float32), outfile)
                np.ndarray.tofile(np.array([numberofWindstreams], dtype=int), outfile)
                np.ndarray.tofile(np.array(lengthofWindStreams, dtype=int), outfile)

                for i in range(numberofWindstreams):
                    currentStream = []
                    for j in range(len(data[i]["y"][0])):
                        currentStream.append(
                            [
                                data[i]["t"][j],
                                data[i]["velocity"][j],
                                data[i]["y"][0][j],
                                data[i]["y"][1][j],
                                data[i]["y"][2][j],
                            ]
                        )

                    np.ndarray.tofile(
                        np.array(currentStream, dtype=np.float32), outfile
                    )
                print(fileName, "saved")
        return self

    def get_velocity(self, t, x, t_0, timedependent):

        t = t if timedependent else t_0
        closest_timeStep = min(self.__timeList, key=lambda x: abs(x - t))
        counter = np.where(self.__timeList == closest_timeStep)[0][0]
        plt_3d_data = self.sim.data_3d[int(counter)]

        mesh = self.sim.meshes[0]
        # Select a quantity
        uvel_idx = plt_3d_data.get_quantity_index("U-VEL")
        vvel_idx = plt_3d_data.get_quantity_index("V-VEL")
        wvel_idx = plt_3d_data.get_quantity_index("W-VEL")
        index_values = self.get_index_values(x)

        # if currentFileName not in self.__memorizedFiles.keys():
        #     self.__memorizedFiles[currentFileName] = self.read_in_bin(currentFileName)
        u_vel_data = plt_3d_data[mesh].data[:, :, :, uvel_idx]
        u_velocity = u_vel_data[index_values[0], index_values[1], index_values[2]]
        v_vel_data = plt_3d_data[mesh].data[:, :, :, vvel_idx]
        v_velocity = v_vel_data[index_values[0], index_values[1], index_values[2]]
        w_vel_data = plt_3d_data[mesh].data[:, :, :, wvel_idx]
        w_velocity = w_vel_data[index_values[0], index_values[1], index_values[2]]

        return np.array([u_velocity, v_velocity, w_velocity])

    def get_index_values(self, x):
        x_index = (x[0] - self.mesh_extent.x_start) / self.__voxalSize["X"]
        if 0 > x_index or x_index > self.mesh.dimension["x"]:
            return np.array([0, 0, 0], dtype=int)
        y_index = (x[1] - self.mesh_extent.y_start) / self.__voxalSize["Y"]
        if 0 > y_index or y_index > self.mesh.dimension["y"]:
            return np.array([0, 0, 0], dtype=int)
        z_index = (x[2] - self.mesh_extent.z_start) / self.__voxalSize["Z"]
        if 0 > z_index or z_index > self.mesh.dimension["z"]:
            return np.array([0, 0, 0], dtype=int)
        return np.array([x_index, y_index, z_index], dtype=int)

    def read_in_bin(self, file):
        with open(file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            _ = np.fromfile(f, dtype=np.float32, count=7)
            (nx, ny, nz) = (header[1], header[2], header[3])
            _ = np.fromfile(f, dtype=np.float32, count=nx * ny * nz)
            data = np.fromfile(f, dtype=np.float32, count=nx * ny * nz * 3)
            data = np.reshape(data, (nx, ny, nz, 3), order="F")
        return data

    def addVelocity(self, oneDataSet):
        """

        :param oneDataSet:
        :return: same dataset with added velocity information
        """

        allSpeeds = [0.0] # all particles start at 0 velocity

        allTimes = oneDataSet["t"]
        allPositions = oneDataSet["y"]
        previousPosition = np.array(
            [allPositions[0][0], allPositions[1][0], allPositions[2][0]]
        )
        for i in range(len(allPositions[0])):
            currentPosition = np.array(
                [allPositions[0][i], allPositions[1][i], allPositions[2][i]]
            )

            deltaTime = allTimes[i] - allTimes[i - 1]
            if i == 0:
                deltaTime = allTimes[i]
            squared_dist = np.sum((currentPosition - previousPosition) ** 2, axis=0)
            dist = np.sqrt(squared_dist)

            speed = dist / deltaTime
            allSpeeds.append(speed)
            previousPosition = currentPosition

        oneDataSet["velocity"] = np.array(allSpeeds)
        return oneDataSet

    def drawPlot(self):

        allData = self.timeReasults
        for time in self.distanceofWindStreams_index.keys():
            data = allData[time]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            for i in self.distanceofWindStreams_index[time]:

                x = data[i]["y"][0][:]
                y = data[i]["y"][1][:]
                z = data[i]["y"][2][:]
                ax.plot(x, y, z)
            ax.set_title( f"Starting Time {time}")
            plt.show()


# %%
def main(args):
    if len(args) < 5:
        print("Need at least 5 arguments")
        print(
            "python windODE_class.py {fds output directory} {fds input file} {start time} {end time} {Wind Start Points}"
        )
        print("Wind Start points can be 1 or more {X_MIN, X_MAX,Y_MIN, Y_MAX}")
        # return
    # dir = args[0]
    # fds_loc = args[1]
    # t_span = [args[2], args[3]]
    # start_points = args[4:]

    fds_loc = "/home/trent/fds3/fds/trails.fds"
    dir = "/home/trent/fds3/"
    fds_loc = "/home/trent/Trunk/Trunk/Trunk.fds"
    dir = "/home/trent/Trunk/"

    fds_loc = "E:\Trunk\Trunk\Trunk\Trunk.fds"
    dir = "E:\Trunk\Trunk\\"

    t_span = [0, 2]
    start_time = time.perf_counter()
    app = windODE(dir, fds_loc, t_span).getStartingPoints()
    # app.startingPointsRibbon([19,1,	3.5],[ 19,	19,	3.5],40)
    # app.readInBin()\
    app.runODE(timedependent=True)
    app.filterOutStreamsByLength()
    app.write2bin("data//temp")
    print(f"Total Time {time.perf_counter()-start_time:0.4f}")
    app.drawPlot()


# %%

if __name__ == "__main__":
    main(sys.argv[1:])

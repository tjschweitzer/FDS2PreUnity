# Import the required modules
import glob
import sys
import time
import os
from collections import defaultdict
import fdsreader as fds
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp

from matplotlib import cm


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
        :vars self.__voxalSize: resolution of each voxal
        :vars self.__maxVelocity: maximum velocity of any particle in the streamlines
        :vars self.startingpoints: list of all starting points to be used in the ODE

        """

        self.sim = fds.Simulation(directory)
        self.fds_input_location = fds_input_location
        self.t_span = t_span
        self.__directory = directory
        self.__qFiles = glob.glob(directory + "*.q")
        self.__timeList = np.array(self.sim.data_3d.times)
        self.__voxalSize = {}
        self.__maxVelocity = 0.0
        self.__maxRe = 0.0
        self.startingpoints = []
        self.__meshBounds = self.sim.meshes[0]
        self.__meshExtent = self.sim.meshes[0].extent

        self.getVoxalSize()

    def getVoxalSize(self):
        """
        Calculates voxal size
        :return:
        """

        self.__voxalSize["vx"] = (self.__meshExtent.x_end - self.__meshExtent.x_start) / (
                self.__meshBounds.dimension["x"] - 1
        )
        self.__voxalSize["vz"] = (self.__meshExtent.z_end - self.__meshExtent.z_start) / (
                self.__meshBounds.dimension["z"] - 1
        )
        self.__voxalSize["vy"] = (self.__meshExtent.y_end - self.__meshExtent.y_start) / (
                self.__meshBounds.dimension["y"] - 1
        )
        return self

    def getStartingPoints(self):

        """
        Creates a list of points  on the outer most voxals of OBSTS, one point per voxal
        :var X_Min_Value center point of minimum x voxal
        :var X_Max_Value center point of maximum x voxal
        :var Y_Min_Value center point of minimum y voxal
        :var Y_Max_Value center point of maximum y voxal

        :return:
        """
        X_Min_Value = (
                self.__meshExtent.x_start + self.__voxalSize["vx"] / 2.0
        )
        X_Max_Value = (
                self.__meshExtent.x_end - self.__voxalSize["vx"] / 2.0
        )
        Y_Min_Value = (
                self.__meshExtent.y_start + self.__voxalSize["vy"] / 2.0
        )
        Y_Max_Value = (
                self.__meshExtent.y_end - self.__voxalSize["vy"] / 2.0
        )
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
                    [XB[0], XB[2], XB[5] + self.__voxalSize["vz"]],
                    [XB[0], XB[3], XB[5] + self.__voxalSize["vz"]],
                    int((XB[1] - XB[0]) / self.__voxalSize["vx"]),
                )

            if XB[0] <= X_Max_Value <= XB[1]:
                self.startingPointsRibbon(
                    [XB[1], XB[2], XB[5] + self.__voxalSize["vz"]],
                    [XB[1], XB[3], XB[5] + self.__voxalSize["vz"]],
                    int((XB[1] - XB[0]) / self.__voxalSize["vx"]),
                )
            if XB[2] <= Y_Min_Value <= XB[3]:
                self.startingPointsRibbon(
                    [XB[0], XB[2], XB[5] + self.__voxalSize["vz"]],
                    [XB[1], XB[2], XB[5] + self.__voxalSize["vz"]],
                    int((XB[3] - XB[2]) / self.__voxalSize["vy"]),
                )
            if XB[2] <= Y_Max_Value <= XB[3]:
                self.startingPointsRibbon(
                    [XB[0], XB[3], XB[5] + self.__voxalSize["vz"]],
                    [XB[1], XB[3], XB[5] + self.__voxalSize["vz"]],
                    int((XB[3] - XB[2]) / self.__voxalSize["vy"]),
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

    def startingPointsRibbon(self, starting_pont, ending_point, number_of_points):
        x_ = np.linspace(starting_pont[0], ending_point[0], number_of_points)
        y_ = np.linspace(starting_pont[1], ending_point[1], number_of_points)
        z_ = np.linspace(starting_pont[2], ending_point[2], number_of_points)
        points = np.stack((x_.flatten(), y_.flatten(), z_.flatten()), axis=1)

        self.startingpoints.extend(points)
        return self

    def runODE(self, timedependent=True):
        self.timeReasults = {}
        for t_start in self.__timeList:
            if t_start > self.t_span[1] or t_start < self.t_span[0]:
                break

            closest_timeStep = min(self.__timeList, key=lambda x: abs(x - t_start))
            counter = np.where(self.__timeList == closest_timeStep)[0][0]
            all_results = []
            for startCounter in range(len(self.startingpoints)):

                y0 = self.startingpoints[startCounter]
                t_span = [
                    min(self.__timeList[counter:]),
                    max(self.__timeList[counter:]),
                ]
                # rtol=1E-4, atol=1E-6,
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
                result_with_re = self.addReynoldsNumber(result_with_velocity)

                current_result_max_re = np.max(result_with_velocity["re"])
                if self.__maxRe < current_result_max_re:
                    print(
                        f"new max Velocity {current_result_max_re} changed from {self.__maxRe}"
                    )
                    self.__maxRe = current_result_max_re
                all_results.append(result_with_re)
            self.timeReasults[t_start] = all_results
        return self

    def write2bin(self, desired_directory, file_name_prefix):
        fileName = os.path.join(desired_directory, file_name_prefix)
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
        u_vel_data = plt_3d_data[mesh].data[:, :, :, uvel_idx]
        u_velocity = u_vel_data[index_values[0], index_values[1], index_values[2]]
        v_vel_data = plt_3d_data[mesh].data[:, :, :, vvel_idx]
        v_velocity = v_vel_data[index_values[0], index_values[1], index_values[2]]
        w_vel_data = plt_3d_data[mesh].data[:, :, :, wvel_idx]
        w_velocity = w_vel_data[index_values[0], index_values[1], index_values[2]]

        return np.array([u_velocity, v_velocity, w_velocity])

    def get_index_values(self, x):
        x_index = (x[0] - self.__meshExtent.x_start) / self.__voxalSize["vx"]
        if 0 > x_index or x_index > self.__meshBounds.dimension["x"]:
            return np.array([0, 0, 0], dtype=int)
        y_index = (x[1] - self.__meshExtent.y_start) / self.__voxalSize["vy"]
        if 0 > y_index or y_index > self.__meshBounds.dimension["y"]:
            return np.array([0, 0, 0], dtype=int)
        z_index = (x[2] - self.__meshExtent.z_start) / self.__voxalSize["vz"]
        if 0 > z_index or z_index > self.__meshBounds.dimension["z"]:
            return np.array([0, 0, 0], dtype=int)
        return np.array([x_index, y_index, z_index], dtype=int)

    def addReynoldsNumber(self, oneDataSet):
        """

        :param oneDataSet:
        :return: same dataset with added velocity information
        """

        allRe = []  # all particles start at 0 velocity

        allTimes = oneDataSet["t"]
        allPositions = oneDataSet["y"]
        for i in range(len(allPositions[0])):
            currentPosition = np.array(
                [allPositions[0][i], allPositions[1][i], allPositions[2][i]]
            )

            currentTime = allTimes[i]
            allRe.append(self.getRaynoldsNumber(currentPosition, currentTime))

        oneDataSet["re"] = np.array(allRe)
        return oneDataSet

    def addVelocity(self, oneDataSet):
        """

        :param oneDataSet:
        :return: same dataset with added velocity information
        """

        allSpeeds = [0.0]  # all particles start at 0 velocity

        allTimes = oneDataSet["t"]
        allPositions = oneDataSet["y"]
        previousPosition = np.array(
            [allPositions[0][0], allPositions[1][0], allPositions[2][0]]
        )
        for i in range(1, len(allPositions[0])):
            currentPosition = np.array(
                [allPositions[0][i], allPositions[1][i], allPositions[2][i]]
            )

            deltaTime = allTimes[i] - allTimes[i - 1]
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
            allRE=[]
            # ax = fig.add_subplot(1, 1, 1, projection="3d")
            for i in self.distanceofWindStreams_index[time]:
                x = data[i]["y"][0][:]
                y = data[i]["y"][1][:]
                z = data[i]["y"][2][:]
                re = data[i]["re"][:]
                allRE.append(re)

            #     ax.plot(x, y, z,  c=cm.viridis((self.__maxRe-re) / self.__maxRe))
            #     #ax.plot(x, y, z, )
            # ax.set_title(f"Starting Time {time}")
            print()
            matplotlib.pyplot.hist(np.array(allRE).flatten(), bins='auto')
            plt.show()

    def compairLines(self):
        allData = self.timeReasults
        for time in self.distanceofWindStreams_index.keys():

            plt.figure(figsize=(12, 8))
            data = allData[time]
            for i in self.distanceofWindStreams_index[time]:
                x = data[i]["y"][0][:]
                y = data[i]["y"][1][:]
                z = data[i]["y"][2][:]
                xy_est = np.polyfit(x, y, 32)
                yz_est = np.polyfit(y, z, 32)
                plt.subplot(2, 1, 1)
                plt.plot(x, y, 'o')
                # evaluate the values for a polynomial
                plt.plot(x, np.polyval(xy_est, x))
                plt.subplot(2, 1, 2)
                plt.plot(y, z, 'o')
                # evaluate the values for a polynomial
                plt.plot(y, np.polyval(yz_est, y))

            plt.tight_layout()
            plt.show()

    def showRE(self):

        allData = self.timeReasults
        for time in self.distanceofWindStreams_index.keys():
            data = allData[time]

            for i in self.distanceofWindStreams_index[time]:
                for j in range(len(data[i]["y"][0][:])):
                    x = data[i]["y"][0][j]
                    y = data[i]["y"][1][j]
                    z = data[i]["y"][2][j]
                    t = data[i]['t'][j]
                    temp = self.getRaynoldsNumber([x, y, z], t)
                    print(f"x {x} -  y {y} - z {z} -  RE {temp}")
            break

    def getRaynoldsNumber(self, x, t):
        closest_timeStep = min(self.__timeList, key=lambda x: abs(x - t))
        counter = np.where(self.__timeList == closest_timeStep)[0][0]
        plt_3d_data = self.sim.data_3d[int(counter)]

        mesh = self.sim.meshes[0]
        # Select a quantity
        try:
            dxeta_idx = plt_3d_data.get_quantity_index("dx/eta")
        except:
            return [0]
        index_values = self.get_index_values(x)

        re_data = plt_3d_data[mesh].data[:, :, :, dxeta_idx]
        re_value = re_data[index_values[0], index_values[1], index_values[2]]

        return re_value


# The Reynolds number is defined as

# Re = uL/ν = ρuL/μ

# where:

# ρ is the density of the fluid (SI units: kg/m3)
# u is the flow speed (m/s)
# L is a characteristic linear dimension (m) (see the below sections of this article for examples)
# μ is the dynamic viscosity of the fluid (Pa·s or N·s/m2 or kg/(m·s))
# ν is the kinematic viscosity of the fluid (m2/s).

# The Dynamic viscocity coefficient is defined as

# μ = μo*(a/b)*(T/To)3/2

# a = 0.555To + C
# b = 0.555T + C

# where

# μ  = viscosity in centipoise at input temperature T
# μ0 = reference viscosity in centipoise at reference temperature To 0.01827
# T   = input temperature in degrees Rankine
# T0 = reference temperature in degrees Rankine 524.07
# C  = Sutherland's constant  = 120


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
    dir = "/home/trent/Trunk/TempCheck"

    fds_loc = "E:\Trunk\Trunk\Trunk\Trunk.fds"
    dir = "E:\Trunk\Trunk\\temp\\"

    t_span = [0, 2]
    start_time = time.perf_counter()
    app = windODE(dir, fds_loc, t_span)
    app.getStartingPoints()
    #app.startingPointsRibbon([19, 1, 3.5], [19, 19, 3.5], 40)
    app.runODE(timedependent=True)
    app.filterOutStreamsByLength()
    # app.write2bin("data","temp")
    print(f"Total Time {time.perf_counter()-start_time:0.4f}")
    app.drawPlot()
    #app.compairLines()


# %%

if __name__ == "__main__":
    main(sys.argv[1:])

# Import the required modules
import sys
from scipy.integrate import solve_ivp
import numpy as np
import glob


class windODE:
    def __init__(self, directory, fds_input_location, t_span, startpoints):

        self.fds_input_location = fds_input_location
        self.startpoints = startpoints
        self.t_span = t_span
        self.__directory = directory
        self.__qFiles = glob.glob(directory + "*.q")

        self.__timeList = np.sort(
            list([self.getFileTimeStep(file) for file in self.__qFiles])
        )
        self.__meshBounds = {}
        self.__maxVelocity = 0.0
        self.fileByTime = {}
        for file in self.__qFiles:
            file_time = self.getFileTimeStep(file)
            if file_time not in self.fileByTime.keys():
                self.fileByTime[file_time] = file
        self.__memorizedFiles = {}

    def readInBin(self):
        print("Reading in File")
        for file in self.__qFiles:
            print(file)
            self.__memorizedFiles[file] = self.read_in_bin(file)
        print("Reading in Done")
        return self

    def getMeshBound(self):
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
                self.voxalSize = {
                    "X": (self.__meshBounds["X_MAX"] - self.__meshBounds["X_MIN"])
                    / self.__meshBounds["I"],
                    "Y": (self.__meshBounds["Y_MAX"] - self.__meshBounds["Y_MIN"])
                    / self.__meshBounds["J"],
                    "Z": (self.__meshBounds["Z_MAX"] - self.__meshBounds["Z_MIN"])
                    / self.__meshBounds["K"],
                }
                break

            lineCounter += 1
        return self

    def getStartingPoints(self):
        self.newStartingPoints = []

        X_Min_Value = (
            self.__meshBounds["X_MIN"] + self.voxalSize["X"] / 2.0
        )  # Center point of mesh
        X_Max_Value = (
            self.__meshBounds["X_MAX"] - self.voxalSize["X"] / 2.0
        )  # Center point of mesh
        Y_Min_Value = (
            self.__meshBounds["Y_MIN"] + self.voxalSize["Y"] / 2.0
        )  # Center point of mesh
        Y_Max_Value = (
            self.__meshBounds["Y_MAX"] + self.voxalSize["Y"] / 2.0
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
            point = [
                (XB[0] + XB[1]) / 2.0,
                (XB[2] + XB[3]) / 2.0,
                (XB[5]) + self.voxalSize["Z"],
            ]
            if "X_MIN" in self.startpoints:
                if XB[0] <= X_Min_Value <= XB[1]:
                    self.newStartingPoints.append(point)
            if "X_MAX" in self.startpoints:
                if XB[0] <= X_Max_Value <= XB[1]:
                    self.newStartingPoints.append(point)
            if "Y_MIN" in self.startpoints:
                if XB[2] <= Y_Min_Value <= XB[3]:
                    self.newStartingPoints.append(point)
            if "Y_MAX" in self.startpoints:
                if XB[2] <= Y_Max_Value <= XB[3]:
                    self.newStartingPoints.append(point)
        print(self.newStartingPoints)
        self.startpoints = self.newStartingPoints
        return self

    def runODE(self,activeTime=True):
        self.timeReasults = {}
        for t_start in self.__timeList:
            if t_start > self.t_span[1] or t_start < self.t_span[0]:
                break

            print(
                f"time {t_start}",
            )

            closest_timeStep = min(self.__timeList, key=lambda x: abs(x - t_start))
            counter = np.where(self.__timeList == closest_timeStep)[0][0]
            all_results = []
            for startCounter in range(len(self.startpoints)):

                y0 = self.startpoints[startCounter]

                t_span = [
                    min(self.__timeList[counter:]),
                    max(self.__timeList[counter:]),
                ]

                result_solve_ivp = solve_ivp(self.get_velocity, t_span, y0, args=[t_span[0],activeTime])
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

    def getFileTimeStep(self, file):
        """
        Extracts the timestep from the filename.

        Parameters:
            file (string): The file name to be processed.

        Returns:
            (Float): Time as a float
        """
        time_sec = int(file.split("_")[-2])
        time_milsec = int(file.split("_")[-1].split(".q")[0])
        return time_sec + time_milsec / 100.0

    def get_velocity(self, t, x,t_0,ActiveTime):

        t= t_0 if ActiveTime else t
        closest_timeStep = min(self.__timeList, key=lambda x: abs(x - t))
        counter = np.where(self.__timeList == closest_timeStep)[0][0]
        currentTimeStep = self.__timeList[counter]
        currentFileName = self.fileByTime[currentTimeStep]
        if currentFileName not in self.__memorizedFiles.keys():
            self.__memorizedFiles[currentFileName] = self.read_in_bin(currentFileName)
        index_values = self.get_index_values(x)
        u_velocity = self.__memorizedFiles[currentFileName][
            index_values[0], index_values[1], index_values[2]
        ][0]
        v_velocity = self.__memorizedFiles[currentFileName][
            index_values[0], index_values[1], index_values[2]
        ][1]
        w_velocity = self.__memorizedFiles[currentFileName][
            index_values[0], index_values[1], index_values[2]
        ][2]
        return np.array([u_velocity, v_velocity, w_velocity])

    def get_index_values(self, x):
        x_index = x[0] - self.__meshBounds["X_MIN"]
        if 0 > x_index or x_index > self.__meshBounds["I"]:
            return np.array([0, 0, 0], dtype=int)
        y_index = x[1] - self.__meshBounds["Y_MIN"]
        if 0 > y_index or y_index > self.__meshBounds["J"]:
            return np.array([0, 0, 0], dtype=int)
        z_index = x[2] - self.__meshBounds["Z_MIN"]
        if 0 > z_index or z_index > self.__meshBounds["K"]:
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
        allSpeeds = [0.0]
        allTime = oneDataSet["t"]
        allPositions = oneDataSet["y"]
        previousPosition = np.array(
            [allPositions[0][0], allPositions[1][0], allPositions[2][0]]
        )
        for i in range(len(allPositions[0])):
            currentPosition = np.array(
                [allPositions[0][i], allPositions[1][i], allPositions[2][i]]
            )
            deltaTime = allTime[i] - allTime[i - 1]
            squared_dist = np.sum((currentPosition - previousPosition) ** 2, axis=0)
            dist = np.sqrt(squared_dist)
            speed = dist / deltaTime
            allSpeeds.append(speed)
            previousPosition = currentPosition

        oneDataSet["velocity"] = np.array(allSpeeds)
        return oneDataSet


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

    fds_loc = "E:\\fds3\\fds\\trails.fds"
    dir = "E:\\fds3\\"

    t_span = [0, 100]
    start_points = ["X_MIN", "X_MAX"]

    app = windODE(dir, fds_loc, t_span, start_points)
    app.getMeshBound().getStartingPoints().readInBin().runODE().write2bin("data\\temp")


# %%

if __name__ == "__main__":
    main(sys.argv[1:])

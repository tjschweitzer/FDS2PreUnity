# %%
# Import the required modules
from scipy.integrate import odeint, solve_ivp

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy

from json import JSONEncoder
import json


# %%


def windODE(directory, startpoints, meshBounds, endtime,fds_input_location):
    def getFileTimeStep(file):
        """
                Extracts the timestep from the filename.

                Parameters:
                    file (string): The file name to be processed.

                Returns:
                    (Float): Time as a float
                """
        time_sec = int(file.split('_')[-2])
        time_milsec = int(file.split('_')[-1].split(".q")[0])
        return time_sec + time_milsec / 100.0

    __directory = directory
    __qFiles = glob.glob(directory + "*.q")

    __timeList = np.sort(list([getFileTimeStep(file) for file in __qFiles]))
    __meshBounds = meshBounds
    __maxVelocity = 0.0
    fileByTime = {}
    for file in __qFiles:
        file_time = getFileTimeStep(file)
        if file_time not in fileByTime.keys():
            fileByTime[file_time] = file
    __memorizedFiles = {}

    def get_velocity(t, x):

        closest_timeStep = min(__timeList, key=lambda x: abs(x - t))
        counter = np.where(__timeList == closest_timeStep)[0][0]
        currentTimeStep = __timeList[counter]
        currentFileName = fileByTime[currentTimeStep]
        if currentTimeStep not in __memorizedFiles.keys():
            __memorizedFiles[currentFileName] = read_in_bin(currentFileName)
        index_values = get_index_values(x)
        u_velocity = __memorizedFiles[currentFileName][index_values[0], index_values[1], index_values[2]][0]
        v_velocity = __memorizedFiles[currentFileName][index_values[0], index_values[1], index_values[2]][1]
        w_velocity = __memorizedFiles[currentFileName][index_values[0], index_values[1], index_values[2]][2]
        return np.array([u_velocity, v_velocity, w_velocity])

    def get_index_values(x):
        x_index = (x[0] - __meshBounds["X_MIN"])
        if 0 > x_index or x_index > __meshBounds["I"]:
            return np.array([0, 0, 0], dtype=int)
        y_index = (x[1] - __meshBounds["Y_MIN"])
        if 0 > y_index or y_index > __meshBounds["J"]:
            return np.array([0, 0, 0], dtype=int)
        z_index = (x[2] - __meshBounds["Z_MIN"])
        if 0 > z_index or z_index > __meshBounds["K"]:
            return np.array([0, 0, 0], dtype=int)
        return np.array([x_index, y_index, z_index], dtype=int)

    def read_in_bin(file):
        with open(file, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            _ = np.fromfile(f, dtype=np.float32, count=7)
            (nx, ny, nz) = (header[1], header[2], header[3])
            _ = np.fromfile(f, dtype=np.float32, count=nx * ny * nz)
            data = np.fromfile(f, dtype=np.float32, count=nx * ny * nz * 3)

            data = np.reshape(data, (nx, ny, nz, 3), order='F')
        return data

    def addVelocity(oneDataSet):
        allSpeeds = [0.0]
        allTime = oneDataSet['t']
        allPositions = oneDataSet['y']
        previousPosition = np.array([allPositions[0][0], allPositions[1][0], allPositions[2][0]])
        for i in range(len(allPositions[0])):
            currentPosition = np.array([allPositions[0][i], allPositions[1][i], allPositions[2][i]])
            deltaTime = allTime[i] - allTime[i - 1]
            squared_dist = np.sum((currentPosition - previousPosition) ** 2, axis=0)
            dist = np.sqrt(squared_dist)
            speed = dist / deltaTime
            allSpeeds.append(speed)
            previousPosition = currentPosition

        oneDataSet['velocity'] = np.array(allSpeeds)
        return oneDataSet

    timeReasults = {}
    for t_start in __timeList:
        print(f"time {t_start}", end=' ')
        if t_start > endtime:
            break
        closest_timeStep = min(__timeList, key=lambda x: abs(x - t_start))
        counter = np.where(__timeList == closest_timeStep)[0][0]
        all_results = []
        for startCounter in range(len(startpoints)):
            print(startCounter, end='  ')
            y0 = startpoints[startCounter]

            t_span = [min(__timeList[counter:]), max(__timeList[counter:])]

            result_solve_ivp = solve_ivp(get_velocity, t_span, y0)
            result_with_velocity = addVelocity(result_solve_ivp)
            current_result_max_vel = np.max(result_with_velocity['velocity'])
            if __maxVelocity < current_result_max_vel:
                print(f"new max Velocity {current_result_max_vel} changed from {__maxVelocity}")
                __maxVelocity = current_result_max_vel
            all_results.append(result_with_velocity)
        timeReasults[t_start] = all_results
        print()

    return timeReasults, __maxVelocity


# %%
dir = "E:\\fds3\\"
start = [[147, 6, 17]]

raw = [[0, 3, 0, 3, 0, 63],
       [3, 6, 0, 3, 0, 63],
       [6, 9, 0, 3, 0, 63],
       [9, 12, 0, 3, 0, 62],
       [12, 15, 0, 3, 0, 62],
       [15, 18, 0, 3, 0, 61],
       [18, 21, 0, 3, 0, 61],
       [21, 24, 0, 3, 0, 60],
       [24, 27, 0, 3, 0, 58],
       [27, 30, 0, 3, 0, 57],
       [30, 33, 0, 3, 0, 56],
       [33, 36, 0, 3, 0, 56],
       [36, 39, 0, 3, 0, 54],
       [39, 42, 0, 3, 0, 53],
       [42, 45, 0, 3, 0, 52],
       [45, 48, 0, 3, 0, 52],
       [48, 51, 0, 3, 0, 51],
       [51, 54, 0, 3, 0, 50],
       [54, 57, 0, 3, 0, 48],
       [57, 60, 0, 3, 0, 47],
       [60, 63, 0, 3, 0, 45],
       [63, 66, 0, 3, 0, 44],
       [66, 69, 0, 3, 0, 44],
       [69, 72, 0, 3, 0, 42],
       [72, 75, 0, 3, 0, 42],
       [75, 78, 0, 3, 0, 40],
       [78, 81, 0, 3, 0, 40],
       [81, 84, 0, 3, 0, 38],
       [84, 87, 0, 3, 0, 36],
       [87, 90, 0, 3, 0, 35],
       [90, 93, 0, 3, 0, 34],
       [93, 96, 0, 3, 0, 33],
       [96, 99, 0, 3, 0, 31],
       [99, 102, 0, 3, 0, 30],
       [102, 105, 0, 3, 0, 29],
       [105, 108, 0, 3, 0, 28],
       [108, 111, 0, 3, 0, 27],
       [111, 114, 0, 3, 0, 25],
       [114, 117, 0, 3, 0, 24],
       [117, 120, 0, 3, 0, 23],
       [120, 123, 0, 3, 0, 22],
       [123, 126, 0, 3, 0, 22],
       [126, 129, 0, 3, 0, 21],
       [129, 132, 0, 3, 0, 20],
       [132, 135, 0, 3, 0, 19],
       [135, 138, 0, 3, 0, 19],
       [138, 141, 0, 3, 0, 18],
       [141, 144, 0, 3, 0, 18],
       [144, 147, 0, 3, 0, 17],
       [147, 150, 0, 3, 0, 16],
       [150, 153, 0, 3, 0, 16],
       [153, 156, 0, 3, 0, 15],
       [156, 159, 0, 3, 0, 14],
       [159, 162, 0, 3, 0, 14],
       [162, 165, 0, 3, 0, 13],
       [165, 168, 0, 3, 0, 12],
       [168, 171, 0, 3, 0, 12],
       [171, 174, 0, 3, 0, 11],
       [174, 177, 0, 3, 0, 11],
       [177, 180, 0, 3, 0, 9],
       [180, 183, 0, 3, 0, 9],
       [183, 186, 0, 3, 0, 8],
       [186, 189, 0, 3, 0, 8],
       [189, 192, 0, 3, 0, 8]]
newStart = [[i[0] + 1.5, i[2], i[-1] + 2] for i in raw]

meshBounds = {
    "X_MIN": 0,
    "Y_MIN": 0,
    "Z_MIN": 0,
    "X_MAX": 192,
    "Y_MAX": 192,
    "Z_MAX": 100,
    "I": 192,
    "J": 192,
    "K": 100}
allResults, maxVel = windODE(dir, newStart[-10:], meshBounds, 120)


# %%

# %%
def plot_everything(oneResult):
    startCounter = 0

    fig = plt.figure(figsize=(6, 6))

    ax = fig.gca(projection='3d')
    for oneResult in allResults:
        ax.plot(oneResult['y'][0, :], oneResult['y'][1, :], oneResult['y'][2, :], color=f'C{startCounter}')
        ax.scatter(oneResult['y'][0, :], oneResult['y'][1, :], oneResult['y'][2, :], color=f'C{startCounter}')

        startCounter += 1

    plt.show()
    startCounter = 0

    plt.figure(figsize=(18, 18))

    plt.subplot(311)
    plt.tight_layout()
    for oneResult in allResults:
        plt.title("X-axis vs Y-axis")
        plt.plot(oneResult['y'][0, :], oneResult['y'][1, :], color=f'C{startCounter}', label=f'{startCounter}')
        plt.scatter(oneResult['y'][0, :], oneResult['y'][1, :], color=f'C{startCounter}')
        startCounter += 1
        # plt.legend()
    startCounter = 0
    plt.subplot(312)
    for oneResult in allResults:
        plt.title("X-axis vs Z-axis")
        plt.plot(oneResult['y'][1, :], oneResult['y'][2, :], color=f'C{startCounter}', label=f'{startCounter}')
        plt.scatter(oneResult['y'][1, :], oneResult['y'][2, :], color=f'C{startCounter}')
        startCounter += 1

    startCounter = 0
    plt.subplot(313)
    plt.tight_layout()
    for oneResult in allResults:
        plt.title("Y-axis vs Z-axis")
        plt.plot(oneResult['y'][0, :], oneResult['y'][2, :], color=f'C{startCounter}', label=f'{startCounter}')
        plt.scatter(oneResult['y'][0, :], oneResult['y'][2, :], color=f'C{startCounter}')
        startCounter += 1

    plt.show()


# %%

# %%
def write2bin(allData, maxVel, fileName):
    for time in allData.keys():
        data = allData[time]
        numberofWindstreams = len(data)
        lengthofWindStreams = [len(x['y'][0]) for x in data]
        print(numberofWindstreams)
        print(lengthofWindStreams)
        time_string = f"{time}".split('.')[1]
        with open(f"{fileName}_{int(time)}_{time_string}.bin", "wb") as outfile:

            np.ndarray.tofile(np.array([maxVel], dtype=np.float32), outfile)
            np.ndarray.tofile(np.array([numberofWindstreams], dtype=np.int), outfile)
            np.ndarray.tofile(np.array(lengthofWindStreams, dtype=np.int), outfile)

            for i in range(numberofWindstreams):
                currentStream = []
                for j in range(len(data[i]['y'][0])):
                    currentStream.append(
                        [data[i]['velocity'][j], data[i]['y'][0][j], data[i]['y'][1][j], data[i]['y'][2][j]])

                np.ndarray.tofile(np.array(currentStream, dtype=np.float32), outfile)
                print(currentStream)

            print(fileName, "saved")


# %%
write2bin(allResults, maxVel, "temp")

# %%

# %%

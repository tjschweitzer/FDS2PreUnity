import os.path
import time
import numpy as np
import glob
import json
import multiprocessing as mp


class fdsOutputToUnity:
    def __init__(self, fds_output_directory, fds_input_location,save_location, saveType="json"):
        self.directory = fds_output_directory
        self.qFiles = glob.glob(fds_output_directory + "*.q")
        self.fileCounter = 0
        self.save_location = save_location
        self.allFilesDict = {}
        self.fds_input_location = fds_input_location

        # self.headerCountTitles = ['smoke', 'U-VELOCITY', 'V-VELOCITY', 'W-VELOCITY', 'fire']
        self.headerCountTitles = []
        self.readInFDS()
        self.lenHeaderCountTitles = len(self.headerCountTitles)

        if len(self.headerCountTitles) == 0:
            print("No Dump line found in FDS input file ")
            return
        self.minValues = np.array([np.inf] * self.lenHeaderCountTitles)
        self.maxValues = np.zeros(self.lenHeaderCountTitles)
        self.save_function = self.write2json if saveType == "json" else self.write2bin

    def readInFDS(self):
        with open(self.fds_input_location) as f:
            lines = f.readlines()

        lineCounter = 0
        while lineCounter < len(lines):
            current_line = lines[lineCounter]
            while "/" not in lines[lineCounter]:
                lineCounter += 1
                current_line = current_line + lines[lineCounter]
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

            lineCounter += 1

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

    def groupFilesbyTime(self):
        """
        Groups filenames by timestamp to allow for multi mesh simulations

        Parameters:
            None

        Returns:
            fileByTime: Dictionary <float,array>
                KEY: Time
                Value: Array of all files with same timestep
        """
        fileByTime = {}
        for file in self.qFiles:
            file_time = self.getFileTimeStep(file)
            if file_time not in fileByTime.keys():
                fileByTime[file_time] = []
            fileByTime[file_time].append(file)
        return fileByTime

    def findMaxValuesParallel(self):
        self.filenames = self.groupFilesbyTime()
        print(self.filenames)
        myMean = []
        myStdDev = []
        pool = mp.Pool()
        for i, returnValue in enumerate(
            pool.imap(self.getValues, self.filenames.keys())
        ):
            print(i, "Done")
            minValue = returnValue[1]
            maxValue = returnValue[2]
            for i in range(self.lenHeaderCountTitles):
                self.maxValues[i] = (
                    self.maxValues[i]
                    if self.maxValues[i] > maxValue[i]
                    else maxValue[i]
                )
                self.minValues[i] = (
                    self.minValues[i]
                    if self.minValues[i] < minValue[i]
                    else minValue[i]
                )
            myMean.append(returnValue[0][0])

        pool.close()
        pool.join()
        self.myMean = np.mean(myMean, axis=0)
        print("My Mean")
        print(self.myMean)
        print("My Min")
        print(self.minValues)
        print("My Max")
        print(self.maxValues)

    def getValues(self, fileTime):
        self.fileCounter += 1
        minValue = [np.inf] * self.lenHeaderCountTitles

        maxValue = [-np.inf] * self.lenHeaderCountTitles

        datamean = []
        for file in self.filenames[fileTime]:
            with open(file, "rb") as f:

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

                data_Min = np.min(data, axis=1)
                # data_Min[0]=1.1
                data_Min[4] = 15

                data_Max = np.max(data, axis=1)
                filePecentile = []
                zeroCounter = 0
                for data_i in data:
                    dataNoZeros = data_i[data_i > data_Min[zeroCounter]]
                    if len(dataNoZeros) < 1:
                        filePecentile.append(0.0)
                        continue
                    filePecentile.append(np.percentile(dataNoZeros, 99))
                    newMin = np.min(dataNoZeros)
                    if newMin < minValue[zeroCounter]:
                        minValue[zeroCounter] = newMin
                    if data_Max[zeroCounter] > maxValue[zeroCounter]:
                        maxValue[zeroCounter] = data_Max[zeroCounter]

                    zeroCounter += 1
            datamean.append(filePecentile)
        return datamean, minValue, maxValue

    def runParallel(self, multiMesh=None):
        """
        The function to read in and save data in parallel.

        Parameters:
            hrrLowerLimit (flaot): The lower bounds for HRRPUA o be save
            smokeLowerLimit (flaot): The lower bounds for HRRPUA o be save
            multiMesh (Dictionary): All infomation needed for multimesh data
                Default : None


        Returns:
            None
        """

        self.hrrLowerLimit = self.minValues[-1]
        self.smokeLowerLimit = self.myMean[0]
        self.multiMesh = multiMesh
        self.filenames = self.groupFilesbyTime()

        pool = mp.Pool()
        for i, temparray in enumerate(
            pool.imap(self.qFileToDict, self.filenames.keys())
        ):
            print(i, temparray)
        pool.close()
        pool.join()
        print(self.minValues)
        print(self.maxValues)

    def getMeshNumber(self, fileName):
        return int(fileName.split("_")[-3])

    def qFileToDict(self, fileTime):

        self.fileCounter += 1
        emptyFile = True
        currentFireArray = []
        currentDensityArray = []
        for file in self.filenames[fileTime]:
            smokeCounter = 0
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

                meshNumber = self.getMeshNumber(file) - 1
                for i in range(nz):
                    for j in range(ny):
                        for k in range(nx):
                            if self.multiMesh is not None:
                                meshRow = meshNumber % (self.multiMesh["I_UPPER"] + 1)
                                meshCol = np.floor(
                                    meshNumber / (self.multiMesh["I_UPPER"] + 1)
                                )
                                meshHeight = np.floor(
                                    meshNumber
                                    / (
                                        (self.multiMesh["I_UPPER"] + 1)
                                        * (self.multiMesh["K_UPPER"] + 1)
                                    )
                                )

                                i += meshCol * self.multiMesh["K"]
                                j += meshHeight * self.multiMesh["J"]
                                k += meshRow * self.multiMesh["I"]

                            if data[counter, 4] >= self.hrrLowerLimit:
                                emptyFile = False
                                pointDict = {
                                    "X": int(i),
                                    "Y": int(j),
                                    "Z": int(k),
                                    "Datum": data[counter, 4],
                                }
                                smokeCounter += 1
                                currentFireArray.append(pointDict)

                            if round(data[counter, 0], 1) < 1.1:
                                emptyFile = False
                                pointDict = {
                                    "X": int(i),
                                    "Y": int(j),
                                    "Z": int(k),
                                    "Datum": data[counter, 0],
                                }

                                currentDensityArray.append(pointDict)
                            counter += 1
        if not emptyFile:
            # a = []
            # for i in currentDensityArray:
            #     a.append(round(i["Datum"],6))
            # a = Counter(a)
            # plt.bar(a.keys(), a.values())
            # plt.show()
            newfile = self.filenames[fileTime][0]
            minMaxDict = {"min": list(self.minValues), "max": list(self.maxValues)}
            dictionary = {
                "fire": currentFireArray,
                "smoke": currentDensityArray,
                "configData": minMaxDict,
            }
            self.save_function(dictionary, newfile)

        return True

    def write2json(self, mydict, fileName):
        newFileName = (
            "_".join(fileName.split("_")[:-3])
            + "_1_"
            + "_".join(fileName.split("_")[-2:])
        )

        with open(f"{newFileName.split('.q')[0]}.json", "w") as outfile:
            json.dump(mydict, outfile)

    def write2bin(self, mydict, fileName):

        newFileName = (
            "_".join(fileName.split("_")[:-3])
            + "_1_"
            + "_".join(fileName.split("_")[-2:])
        )
        newFileName = newFileName.split(".q")[0] + ".bin"
        newFileName = self.save_location+os.path.basename(newFileName)
        headerCountTitles = ["smoke", "U-VELOCITY", "V-VELOCITY", "W-VELOCITY", "fire"]
        # header {number of eachtype of value to be saved  } 'DENSITY','U-VELOCITY','V-VELOCITY','W-VELOCITY','HRRPUV'

        header = np.array(
            [
                len(mydict[title]) if title in mydict else 0
                for title in headerCountTitles
            ],
            dtype=np.int,
        )
        print(fileName, header)

        with open(f"{newFileName.split('.q')[0]}", "wb") as outfile:

            np.ndarray.tofile(header, outfile)

            allData = []
            for i in range(self.lenHeaderCountTitles):
                h = headerCountTitles[i]

                if h not in mydict:
                    continue
                for j in range(header[i]):
                    point = mydict[h][j]
                    allData.append(point["X"])
                    allData.append(point["Y"])
                    allData.append(point["Z"])

                    allData.append(point["Datum"])

            print("min", self.minValues)
            print("max", self.maxValues)
            np.ndarray.tofile(np.array(self.minValues, dtype=np.float32), outfile)
            np.ndarray.tofile(np.array(self.maxValues, dtype=np.float32), outfile)
            np.ndarray.tofile(np.array(allData, dtype=np.float32), outfile)
            print(fileName, "saved")


# print( data, header[1:-1])
if __name__ == "__main__":
    startTime = time.time()
    app = fdsOutputToUnity(
        "E:\\fds3\\", "E:\\fds3\\fds\\trails.fds", "bin"
    )
    meshDict = {
        "I_UPPER": 7,
        "J_UPPER": 0,
        "K_UPPER": 4,
        "I": 128,
        "J": 164,
        "K": 40,
    }
    app.findMaxValuesParallel()
    app.runParallel()
    print(time.time() - startTime)

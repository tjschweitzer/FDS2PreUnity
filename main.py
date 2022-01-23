import sys, shutil
from fds2ComplexGeom import fds2ComplexGeom
from fdsOutput2Unity import fdsOutputToUnity
from windODE_class import windODE

import os


def main(args):
    fdsOutputDir = "E:\\fds3\\"  # directory of the plot3d files from fds output
    fdsInputFile = "E:\\fds3\\fds\\trails.fds"  # location of fds input file
    saveLocation = "E:\\saveLoc\\"  # desired directory for all files to be saved

    # windODE Variables
    t_span = [0, 100]  # range of start time for wind vectors
    starting_points = ["X_MIN", "X_MAX", "Y_MIN", "Y_MAX"]  # what sides of mesh used

    if len(args) > 6:
        fdsOutputDir = args[0]  # directory of the plot3d files from fds output
        fdsInputFile = args[1]  # location of fds input file
        saveLocation = args[2]  # desired directory for all files to be saved

        # windODE Variables
        t_span = args[3:5]  # range of start time for wind vectors
        starting_points = args[5:]  # what sides of mesh used

    # Creating the directories for where all the custom data will be saved
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    if not os.path.exists(os.path.join(saveLocation, "wind")):
        os.makedirs(os.path.join(saveLocation, "wind"))
    if not os.path.exists(os.path.join(saveLocation, "fds")):
        os.makedirs(os.path.join(saveLocation, "fds"))

    # Copies the fds input file to save location
    shutil.copy(fdsInputFile, os.path.join(saveLocation, "fds"))

    # Converts plot3d data to a sparce matrix binary file
    app = fdsOutputToUnity(fdsOutputDir, fdsInputFile, saveLocation, "bin")
    app.findMaxValuesParallel()
    app.runParallel()

    # Ordinary Differential Equations for wind vectors
    windODE(fdsOutputDir, fdsInputFile, t_span, starting_points).getMeshBound().getStartingPoints().readInBin().runODE().write2bin(
        os.path.join(os.path.join(saveLocation, "wind"), "temp")
    )

    # converts fds input file into a complex geometry json
    app = fds2ComplexGeom(fdsInputFile)
    app.save2Json(os.path.join(os.path.join(saveLocation, "fds"), "topo.json"))


if __name__ == "__main__":
    main(sys.argv[1:])

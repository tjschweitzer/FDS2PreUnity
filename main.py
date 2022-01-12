# This is a sample Python script.


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys, shutil

import fds2ComplexGeom
from fdsOutput2Unity import fdsOutputToUnity
import windODE_class
import os


def main(args):
    # fdsOutputDir = args[0] # directory of the plot3d files from fds output
    # fdsInputFile = args[1] # location of fds input file
    # saveLocation = args[2] # desired directory for all files to be saved
    fdsOutputDir = "E:\\fds3\\" # directory of the plot3d files from fds output
    fdsInputFile = "E:\\fds3\\fds\\trails.fds" # location of fds input file
    saveLocation = "E:\\saveLoc\\" # desired directory for all files to be saved

    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    if not os.path.exists(os.path.join(saveLocation,"wind")):
        os.makedirs(os.path.join(saveLocation,"wind"))
    if not os.path.exists(os.path.join(saveLocation, "fds")):
        os.makedirs(os.path.join(saveLocation, "fds"))

    # app = fdsOutputToUnity(
    #     fdsOutputDir, fdsInputFile,saveLocation, "bin"
    # )
    # app.findMaxValuesParallel()
    # app.runParallel()

    app = windODE_class.windODE(fdsOutputDir,fdsInputFile,[0,100],["X_MIN","X_MAX","Y_MIN","Y_MAX",])
    app.getMeshBound().getStartingPoints().readInBin().runODE().write2bin(os.path.join(os.path.join(saveLocation,"wind"),"temp"))
    shutil.copy(fdsInputFile,os.path.join(saveLocation,"fds"))
    app = fds2ComplexGeom.fds2ComplexGeom(fdsInputFile)
    app.save2Json(os.path.join(os.path.join(saveLocation,"fds"),"sample.json"))

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[1:])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

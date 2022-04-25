import shutil
import sys
from fds2ComplexGeom import fds2ComplexGeom
from fdsOutput2Unity import fdsOutputToUnity
from FdsPathLines import FdsPathLines

import os


def main(args):
    fdsOutputDir = (
        "/home/trent/Trunk/TimeDelay"  # directory of the plot3d files from fds output
    )
    fdsInputFile = "/home/trent/Trunk/Trunk/Trunk.fds"  # location of fds input file
    saveLocation = "/home/trent/Test"  # desired directory for all files to be saved

    tree_id = "Generic Foliage"
    non_terrain_obsts = ["Trunk"]
    if len(args) > 4:
        fdsOutputDir = args[0]  # directory of the plot3d files from fds output
        fdsInputFile = args[1]  # location of fds input file
        saveLocation = args[2]  # desired directory for all files to be saved

        # complexGeom Variables
        tree_id = args[3]  # Label ID for Trees
        non_terrain_obsts = args[
            4:
        ]  # List of any labels of non terrain non tree objects

    # Creating the directories for where all the custom data will be saved
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    if not os.path.exists(os.path.join(saveLocation, "wind")):
        os.makedirs(os.path.join(saveLocation, "wind"))
    if not os.path.exists(os.path.join(saveLocation, "fds")):
        os.makedirs(os.path.join(saveLocation, "fds"))

    # Copies the fds input file to save location
    shutil.copy(fdsInputFile, os.path.join(saveLocation, "fds"))

    # converts fds input file into a complex geometry json
    app_geom = fds2ComplexGeom(fdsInputFile, tree_id, non_terrain_obsts)
    app_geom.save_to_json(os.path.join(os.path.join(saveLocation, "fds"), "topo.json"))

    # Ordinary Differential Equations for wind vectors

    ode_app = FdsPathLines(fdsOutputDir, fdsInputFile)
    ode_app.set_turbulent_laminar_poi()
    ode_app.start_ode(True)
    ode_app.filter_streams_by_length()
    ode_app.draw_stream_lines()
    ode_app.write_h5py(os.path.join(saveLocation, "wind"), "weightedMeans")

    # Converts plot3d data to a sparce matrix binary file
    app_hrr = fdsOutputToUnity(fdsOutputDir, fdsInputFile, saveLocation, "bin")
    app_hrr.findMaxValuesParallel()
    app_hrr.runParallel()


if __name__ == "__main__":
    main(sys.argv[1:])

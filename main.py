# This is a sample Python script.


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys

import fds2ComplexGeom
from fdsOutput2Unity import fdsOutputToJson
import windODE_class

def main(args):
    fdsOutputDir = args[0]
    fdsInputFile = args[1]
    fdsout = fdsOutputToJson(fdsOutputDir,fdsInputFile,'bin')

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[1:])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

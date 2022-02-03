import glob
import numpy as np
fileDir = "/home/trent/TrunkAGL/"
files = glob.glob(fileDir+"*.sf")
for file in files:
    with open(file, "rb") as f:

        counter = 0
        header = np.fromfile(f, dtype=np.float64, count=5)
        _ = np.fromfile(f, dtype=np.float32, count=7)
        print(header)
        print(_)
        try:
            while True:
                temp = np.fromfile(f, dtype=np.float32, count=1)
                print(temp)
        except:
            print("oops")
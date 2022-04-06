import h5py

import fdsreader as fds

# fileName = "C:\\FDS2UnityTool\\data\\weightedMeans_1_0.hdf5"
#
# f = h5py.File(fileName,  "r")
#
# for item in f.keys():
#
#     print(item + ":", f[item])
#
#
#
#
# f.close()

sim = fds.Simulation("E:\Trunk\\Trunk\\StableWind")
# sim.geom.GeometryCollection.GeometryCollection(*geom_boundaries: Iterable[fdsreader.geom.geometry.GeomBoundary])
print()

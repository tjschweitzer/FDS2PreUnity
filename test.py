import h5py


fileName = "C:\\FDS2UnityTool\\data\\weightedMeans_1_0.hdf5"

f = h5py.File(fileName,  "r")

for item in f.keys():

    print(item + ":", f[item])




f.close()
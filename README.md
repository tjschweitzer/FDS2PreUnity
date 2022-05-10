# FDS2UnityTool
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



FdsPathLines
----------------------
Reads in FDS output files and outputs hdf5 files containing (N) pathlines sampled around highest reynolds values. where N is 1/2 (MeshVolume)^()/  

Command line example
```bash
python FdsPathLines.py {FDS Output Folder} {FDS Input File Path} {Output Directory} {filename prefix}
```

python example
```python
import FdsPathLines
fds_output_dir = "/Example/path" # path to FDS output Folder
fds_input_loc = "/Example/fds/inputfile.fds" # path to FDS output Folder
hdf5_output_dir = "/Example/outputdirectoy"
output_fileprefix = "myExableFiles"

# initalizes the fds data 
app = FdsPathLines(fds_output_dir, fds_input_loc)

# Selects the seed points for simulation
app.set_turbulent_laminar_poi()

# Other options for seed placement
# app.set_even_distro_poi()
# app.set_random_distro_poi()

# Runs RK4(5) ODE 
app.StartODE()

# Plots line graphs of each timeset {optional command}
app.draw_stream_lines()

# saves file to desired dir with filename structure {prefix}_1_{timestep}.hdf5
app.write_h5py(hdf5_output_dir, output_fileprefix)

```


fds2ComplexGeom
----------------------
Reads in FDS  input file and generates information for a 3-D mesh visualized in Unity and a list of all tree locations,  tree stat
s  
FDS terrain must be in an evenly distributed system

Command line example
```bash
python fds2ComplexGeom.py {FDS Input File Path} {Output Directory} {tree Label} {Non terrain object Label(s)}
```

python example
```python
import fds2ComplexGeom
fds_input_loc = "/Example/fds/inputfile.fds" # path to FDS output Folder
output_dir = "/Example/outputdirectoy"

# initializes the fds data 
app = fds2ComplexGeom(fds_input_loc)

# saves data to JSon File
app.save2Json(output_dir)

```

fdsComplexGeom
----------------------
Reads in FDS  input file and generates hrrpuv location information, then is output to json bin or hdf5

Command line example
```bash
python fdsOutput2Unity.py {FDS Output Directory} {FDS Input File Path} {Output Directory} {Ouput FileType}
```

python example

```python
import fdsOutput2Unity

fds_output_dir = "/Example/path"  # path to FDS output Folder
fds_input_loc = "/Example/fds/inputfile.fds"  # path to FDS output Folder
output_dir = "/Example/outputdirectoy"
output_filetype = "myExampleFiles"

# initalizes the fds data bl

app = fdsOutput2Unity(
    fds_output_dir, fds_input_loc, output_dir, output_filetype
)

# Finds the max HRRPUV for the simulation
app.find_max_values_parallel()

# Saves the location and value for each voxel in the top 99% of the timestep
app.runParallel()

```
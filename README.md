# OpenPMD creator

Some python functions to generate openPMD files which can be used to initialse particles in WarpX simulations. Note that depending on the size of the simulation and number of particles per cell the particle files (and RAM usage) can be quite high (> tens of GB). The code could prehaps be written more efficently but it should do the job. In my opinion it is best to create the files on whatever server you will run the WarpX simulation on. Note: even this small example hdf5 file is over a GB.

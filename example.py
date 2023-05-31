import numpy as np

import openPMD_particlefile_creator as pfc


def main(debug=False):
    # These values are just for illustration
    xmin, xmax, nx = (0, 20e-6, 1000)
    zmin, zmax, nz = (0, 20e-6, 1000)
    dx = (xmax - xmin) / nx
    dz = (zmax - zmin) / nz

    # density in /m3
    density = np.fromfile("example.bin").reshape(1000, 1000) * 1e6
    print(density.shape)
    # get X and Z position values
    # sometimes with floats xmax is included
    # the index at the end removes this error
    x_left = np.arange(xmin, xmax, dx)[:nx]
    z_left = np.arange(zmin, zmax, dz)[:nz]

    X, Z = np.meshgrid(x_left, z_left)

    # ~ 1 particle per critical density
    ppc_per_density = 1 / 1e27
    min_ppc = 1
    max_ppc = 100
    ppc_grid = pfc.calculate_number_ppc(density, ppc_per_density, min_ppc, max_ppc)
    print(density.shape, ppc_grid.shape)
    # cell volume is dx * dy * dz. In 2D dy is 1m
    weight_grid = pfc.calculate_macroparticle_weight(density, (dx * dz * 1), ppc_grid)
    weight_arr = pfc.create_weight_array(weight_grid, ppc_grid)

    print(density.shape, ppc_grid.shape, weight_grid.shape)
    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1)
        ax[0].set(aspect=1)
        ax[1].set(aspect=1)
        im = ax[0].pcolormesh(x_left, z_left, ppc_grid)
        fig.colorbar(im, ax=ax[0])
        im = ax[1].pcolormesh(x_left, z_left, weight_grid / weight_grid.max())
        fig.colorbar(im, ax=ax[1])
        plt.show()

    # now get the positions
    pos_arr = pfc.create_position_array(X, Z, ppc_grid, dx, dz)

    # now I need to output the values to a file
    species = "electrons"
    with pfc.hdf5_species_template("example.h5", species) as hdf5_file:
        pfc.set_weight(hdf5_file, species, weight_arr)
        pfc.set_position(hdf5_file, species, pos_arr, geom="2D")
        mom_arr = np.zeros(pos_arr.shape)
        pfc.set_momentum(hdf5_file, species, mom_arr)
        pfc.set_mass(hdf5_file, species, 9.11e-31)
        pfc.set_charge(hdf5_file, species, -1)

    return


if __name__ == "__main__":
    main(debug=True)

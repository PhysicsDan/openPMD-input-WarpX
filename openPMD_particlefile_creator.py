"""
Functions to create openPMD file for WarpX particle initialisation.

The functions can:
- Convert a number density cell into evenly distributed macro particles
- Write them to an HDF5 file

To do this you need to:
Take a grid of density and associated positions and convert thsi into weight
and XYZ positions.

The weight array is just the flattened density array (x some scale factor)
with the element value repeated for each item in the array.

The positions correspond to the position of each cell. The providing the
number of particles in each direction you get the positions for of each
macroparticle.


WARNING: This code only works for the same number of particles per cell
in each direction and in 2D cartesian. I will update the code if I need otherwise. Feel free to fork the repository.
"""
import h5py
import numpy as np


def evenly_spaced_values(x0: float, x1: float, nx: int):
    """
    Return an array of evenly spaced values

    Desc: Returns an array of nx evenly spaced values between
    x0 and x1. The 1st and last values are 1/2 a step from the
    bounds.

    x0  (float)     lower bound of range
    x1  (float)     upper bound of range
    nx  (int)       number of points in the range
    """
    if x1 <= x0:
        raise ValueError("x1 must be greater than x0")
    dx = (x1 - x0) / nx
    values = np.linspace(x0, x1 - dx, nx) + dx / 2

    return values


def calculate_positions(x0: float, z0: float, dx: float, dz: float, nx: int, nz: int):
    "Return arrays for x and y corresponding to evenly spaced particle positions"
    x_spaced = evenly_spaced_values(x0, x0 + dx, nx)
    z_spaced = evenly_spaced_values(z0, z0 + dz, nz)

    X, Z = np.meshgrid(x_spaced, z_spaced)

    return X.flatten(), Z.flatten()


def calculate_macroparticle_weight(density, cell_volume, ppc):
    """
    Return the value of macroparticle weight of a cell

    args:
    density     (numpy.ndarray)     Array of number density values
    cell_volume (float)             Cell volume (dx * dy * dz)
    ppc         (numpy.ndarray)     Array containing number of macro particle per cell

    returns:
    macroparticle_weight (np.ndarray)   Array same shape as density with the weight of the\
                                        macro particles in each cell 
    """

    number_real_particles = density * cell_volume
    macroparticle_weight = number_real_particles / ppc
    return macroparticle_weight


def calculate_number_ppc(density, ppc_per_density, min_density=0, max_density=np.inf):
    """
    Returns array of ppc

    args:
    density         (numpy.ndarray)     Array of number density values
    ppc_per_density (float)             ppc ~ density/ppc_per_density

    Note: currently asumes a square cell. Will be updated when needed...
    """
    density = np.clip(density, min_density, max_density)
    ppc = density / ppc_per_density
    ppc = round_to_nearest_square(ppc)
    return ppc


def round_to_nearest_square(arr):
    """
    Rounds each element in the input numpy array to the nearest square number.

    Parameters:
        arr (numpy.ndarray): The input numpy array.

    Returns:
        numpy.ndarray: The rounded array, where each element is the nearest square number.
    """
    sqrt_arr = np.sqrt(arr)
    rounded_arr = np.round(sqrt_arr)
    squared_arr = rounded_arr**2
    return squared_arr.astype(int)


def create_weight_array(particle_weight_grid, ppc_grid):
    particle_weight = particle_weight_grid.flatten()
    ppc = ppc_grid.flatten()
    weight_array = np.repeat(particle_weight, ppc)
    return weight_array


def create_position_array(x_mesh, z_mesh, ppc_grid, dx, dz):
    """
    Return arrays for macroparticle position

    Parameters:
        x_mesh (numpy.ndarray) Meshgrid of X positions
        z_mesh (numpy.ndarray) Meshgrid of Z positions
        ppc_grid (numpy.ndarray) Grid of macroparticles per cell
        dx (float) Cell size in x direction
        dz (float) Cell size in z direction
    """

    # flatten the mesh
    ppc_grid = ppc_grid.flatten()
    # here I will remove any zero weight cells to speed up for loop
    non_zero_idx = ppc_grid > 0
    ppc_grid = ppc_grid[non_zero_idx]

    # flatten the grids for conveniance and remove zeros
    x_mesh = x_mesh.flatten()[non_zero_idx]
    z_mesh = z_mesh.flatten()[non_zero_idx]

    # create output arrays
    x_out = np.zeros(ppc_grid.sum())
    y_out = x_out.copy()
    z_out = x_out.copy()

    # keep track of where you are
    start_idx = 0

    for i in range(ppc_grid.size):
        # get x, z and ppc value for cell
        x_0 = x_mesh[i]
        z_0 = z_mesh[i]
        ppc = ppc_grid[i]

        # get the macroparticle postions in x and z
        x_ppc = evenly_spaced_values(x_0, x_0 + dx, int(np.sqrt(ppc)))
        z_ppc = evenly_spaced_values(z_0, z_0 + dz, int(np.sqrt(ppc)))
        # get a meshgrid with size ppc
        X, Z = np.meshgrid(x_ppc, z_ppc)

        # store the flattened positons in the array
        x_out[start_idx:ppc] = X.flatten()
        z_out[start_idx:ppc] = Z.flatten()
        start_idx += ppc

    return x_out, y_out, z_out


#################################
# -------------------------------#
# The code below is for writing to the hdf5 file
# As of May '22 this works but the WarpX developers
# seem to be making changes so it may break. It should
# be relatively easy to modify the code below as needed


def hdf5_species_template(filename, species):
    """
    This function returns a h5py file with a template that can be used as a particle input

    Use this function as...
    with hdf5_species_template(filename) as f:
        "insert any code you want here"
    """
    f = h5py.File(filename, "w")
    f.attrs["openPMD"] = "1.1.0"
    f.attrs["openPMDextension"] = np.uint32(1)  # for ED-PIC
    f.attrs["basePath"] = "/data/%T/"
    f.attrs["iterationEncoding"] = "groupBased"
    f.attrs["iterationFormat"] = "/data/%T/"
    f.attrs["author"] = "Dan"  # "Daniel Molloy <dmolloy09@qub.ac.uk>"
    # f.attrs["meshesPath"] = "meshes/"
    f.attrs["particlesPath"] = "particles/"
    # create a group
    # you need a subgroup for each particle species
    # the general structure I have seen for particles is...
    # data/<file_no>/particles/<species>/position/
    # data/<file_no>/particles/<species>/momentum/
    pos = f.create_group(f"/data/0/particles/{species}/position/")
    posoff = f.create_group(f"/data/0/particles/{species}/positionOffset/")
    mom = f.create_group(f"/data/0/particles/{species}/momentum/")
    # pos_off = f.create_group(f"/data/1/particles/{species}/positionOffset/")

    f["/data/0/"].attrs["dt"] = 1.0
    f["/data/0/"].attrs["time"] = 0.0
    f["/data/0/"].attrs["timeUnitSI"] = 1.0

    # set the attributes required
    pos.attrs["unitDimension"] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mom.attrs["unitDimension"] = [1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    pos.attrs["timeOffset"] = 0.0
    mom.attrs["timeOffset"] = 0.0
    posoff.attrs["unitDimension"] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    posoff.attrs["timeOffset"] = 0.0

    return f


def set_weight(f, species, weights):
    # create a dataset for weight
    wgt = f.create_dataset(f"data/0/particles/{species}/weighting", data=weights)
    wgt.attrs["unitDimension"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    wgt.attrs["unitSI"] = 1.0
    wgt.attrs["weightingPower"] = 1.0
    wgt.attrs["macroWeighted"] = np.uint32(1)
    wgt.attrs["timeOffset"] = 0.0


def set_position(f, species, pos_arr, geom="2D"):
    if geom == "2D":
        geom = ["x", "z"]
    for idx, axis in enumerate(geom):
        pos = f.create_dataset(
            f"data/0/particles/{species}/position/{axis}", data=pos_arr[idx, :]
        )
        pos.attrs["unitSI"] = 1.0
    f[f"data/0/particles/{species}/position/"].attrs["macroWeighted"] = np.uint32(0)
    f[f"data/0/particles/{species}/position/"].attrs["weightingPower"] = np.float64(1)

    if geom == "2D":
        geom = ["x", "z"]
    for idx, axis in enumerate(geom):
        pos = f.create_dataset(
            f"data/0/particles/{species}/positionOffset/{axis}",
            data=np.zeros(pos_arr[idx, :].shape),
        )
        pos.attrs["unitSI"] = 1.0
    f[f"data/0/particles/{species}/positionOffset/"].attrs["macroWeighted"] = np.uint32(
        0
    )
    f[f"data/0/particles/{species}/positionOffset/"].attrs[
        "weightingPower"
    ] = np.float64(1)


def set_momentum(f, species, mom_arr, geom="2D"):
    f[f"data/0/particles/{species}/momentum/"].attrs["macroWeighted"] = np.uint32(0)
    if geom == "2D":
        geom = ["x", "z"]
    for idx, axis in enumerate(geom):
        mom = f.create_dataset(
            f"data/0/particles/{species}/momentum/{axis}", data=mom_arr[idx, :]
        )
        mom.attrs["unitSI"] = 1.0
    f[f"data/0/particles/{species}/momentum/"].attrs["macroWeighted"] = 0
    f[f"data/0/particles/{species}/momentum/"].attrs["weightingPower"] = np.float64(1)


def set_mass(f, species, mass_kg: float):
    m = f.create_group(f"/data/0/particles/{species}/mass/")

    m.attrs["value"] = mass_kg
    m.attrs["unitDimension"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    m.attrs["unitSI"] = 1.0


def set_charge(f, species, charge: float):
    q = f.create_group(f"/data/0/particles/{species}/charge/")

    q.attrs["value"] = charge
    q.attrs["unitDimension"] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    q.attrs["unitSI"] = 1.0

import atomap.api as am
import hyperspy.api as hs
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
import matplotlib.pyplot as plt

def reconstruct_via_atomap(image, separation, refine=True):
    """
    Reconstruct real-space lattice from atomic columns via
    atomap Python library.
    
    Parameters:
    
    image: numpy array
        Input image
        
    separation: int
        minimum pixel separation of atomic columns
    refine: bool, default=True
        apply additional refinement procedure on 
        top of reconstructed lattice, as implemented
        in atomap
        
    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>, Andreas Leitherer <andreas.leitherer@gmail.com>
    
    """

    image = hs.signals.Signal2D(image)

    separation_range = (separation - 1, separation + 1)
    
    peaks = am.get_feature_separation(image, separation_range=separation_range)
    atom_positions = am.get_atom_positions(image, separation=separation)
    
    
    if refine:
        
        min_peak_separation = separation
        tr_img = hs.signals.Signal2D(image)

        s_peaks = peaks #am.get_feature_separation(hs.signals.Signal2D(image),
                  #                          separation_range=(min_peak_separation, min_peak_separation * 2),
                  #                          show_progressbar=False)
        # Get peak positions and determine sublattice
        peak_pos = am.get_atom_positions(tr_img, separation=min_peak_separation)
        peak_pos = am.Sublattice(peak_pos, image=tr_img.data)

        # Refine peak positions using center of mass and 2D Gaussians based on NN distance
        peak_pos.find_nearest_neighbors()
        peak_pos.refine_atom_positions_using_center_of_mass()
        peak_pos.refine_atom_positions_using_2d_gaussian()

        #peak_pos.plot(navigator='signal')

        # Covert peaks to array
        peak_list = peak_pos.atom_list
        num_peaks = np.shape(peak_list)
        num_peaks = num_peaks[0]

        peaks = np.zeros((num_peaks,2))
        for i in range(0, num_peaks):
            peaks[i,:] = [peak_list[i].pixel_x, peak_list[i].pixel_y]
        atom_positions = peaks
    
    return atom_positions





def get_nn_distance(atoms, distribution='quantile_nn', cutoff=20.0,
                    min_nb_nn=1,#5,
                    pbc=True, plot_histogram=False, bins=100, 
                    constrain_nn_distances=False, nn_distances_cutoff=0.9, 
                    element_sensitive=False, central_atom_species=26, neighbor_atoms_species=26,
                    return_more_nn_distances=False, return_histogram=False):
    """
    Function for scaling given lattice (as defined by ASE Atoms object)
    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>, Andreas Leitherer <andreas.leitherer@gmail.com>
    """
    
    if not pbc:
        atoms.set_pbc((False, False, False))

    nb_atoms = atoms.get_number_of_atoms()
    cutoffs = np.ones(nb_atoms) * cutoff
    # Notice that if get_neighbors(a) gives atom b as a neighbor,
    #    then get_neighbors(b) will not return a as a neighbor - unless
    #    bothways=True was used."
    nl = NeighborList(cutoffs, skin=0.1, self_interaction=False, bothways=True)
    # nl.build(atoms) previously used.
    nl.update(atoms)
    nn_dist = []

    for idx in range(nb_atoms):
        # element sensitive part - only select atoms of specified chemical species as central atoms
        if element_sensitive:
            if atoms.get_atomic_numbers()[idx]==central_atom_species:
                pass
            else:
                continue        
        
        #print("List of neighbors of atom number {0}".format(idx))
        indices, offsets = nl.get_neighbors(idx)
        if len(indices) >= min_nb_nn: # before was >!!
            coord_central_atom = atoms.positions[idx]
            # get positions of nearest neighbors within the cut-off
            dist_list = []
            for i, offset in zip(indices, offsets):
                # element sensitive part - only select neighbors of specified chemical species
                if element_sensitive:
                    if atoms.get_atomic_numbers()[i]==neighbor_atoms_species:
                        pass
                    else:
                        continue
                # center each neighbors wrt the central atoms
                coord_neighbor = atoms.positions[i] + np.dot(offset, atoms.get_cell())
                # calculate distance between the central atoms and the neighbors
                dist = np.linalg.norm(coord_neighbor - coord_central_atom)
                dist_list.append(dist)

            # dist_list is the list of distances from the central_atoms
            if len(sorted(dist_list)) > 0:
                # get nearest neighbor distance
                nn_dist.append(sorted(dist_list)[0])
            else:
                print("List of neighbors is empty for some atom. Cutoff must be increased.")
                return None
        else:
            print("Atom {} has less than {} neighbours. Skipping.".format(idx, min_nb_nn))


    if constrain_nn_distances:
         original_length = len(nn_dist)
         # Select all nearest neighbor distances larger than nn_distances_cutoff
         threshold_indices = np.array(nn_dist) > nn_distances_cutoff 
         nn_dist = np.extract(threshold_indices , nn_dist)
         if len(nn_dist)<original_length:
             print("Number of nn distances has been reduced from {} to {}.".format(original_length,len(nn_dist)))

    if distribution == 'avg_nn':
        length_scale = np.mean(nn_dist)
    elif distribution == 'quantile_nn':
        # get the center of the maximally populated bin
        hist, bin_edges = np.histogram(nn_dist, bins=bins, density=False)

        # scale by r**2 because this is how the rdf is defined
        # the are of the spherical shells grows like r**2
        hist_scaled = []
        for idx_shell, hist_i in enumerate(hist):
            hist_scaled.append(float(hist_i)/(bin_edges[idx_shell]**2))

        length_scale = (bin_edges[np.argmax(hist_scaled)] + bin_edges[np.argmax(hist_scaled) + 1]) / 2.0

        if plot_histogram:
            # this histogram is not scaled by r**2, it is only the count
            plt.hist(nn_dist, bins=bins)  # arguments are passed to np.histogram
            plt.title("Histogram")
            plt.show()
    else:
        raise ValueError("Not recognized option for atoms_scaling. "
                         "Possible values are: 'min_nn', 'avg_nn', or 'quantile_nn'.")
                         
    if return_more_nn_distances and distribution=='quantile_nn':
        length_scale_3 = (bin_edges[np.argsort(hist_scaled)[-3:][0]] + bin_edges[np.argsort(hist_scaled)[-3:][0] + 1]) / 2.0
        length_scale_2 = (bin_edges[np.argsort(hist_scaled)[-3:][1]] + bin_edges[np.argsort(hist_scaled)[-3:][1] + 1]) / 2.0
        return length_scale, length_scale_2, length_scale_3
    elif return_histogram:
        return length_scale, hist_scaled, nn_dist
    else:
        return length_scale


def norm_window_lattice(atomic_columns, reference_lattice, 
                        window_size, pixel_to_angstrom):
    """
    Helper function for normalizing input lattice, 
    where normalization is performed with respect to a reference lattice.
    The input lattice is shifted to the origin and 
    then a radial mask is iteratively applied, 
    cropping the input lattice until it contains 
    the same number of coordinates as the reference lattice.

    Parameters
    ----------
    atomic_columns : ndarray
        2D numpy array containing real space 2D lattice
        coordinates reconstructed from image 
        (i.e., the atomic column positions).
    reference_lattice : ndarray
        2D numpy array containtin the 2D reference lattice 
        coordinates. This lattice is used as reference. 
    window_size : float
        Window size employed, in units of pixels.
    pixel_to_angstrom : float
        Relation between pixel and Angstrom, i.e.,
        the amount of Angstrom corresponding to one pixel.

    Returns
    -------
    lattice : ndarray
        Normed lattice.

    """
    
    # Select box
    peaks_box = np.array(atomic_columns)
  
    # Shift to origin
    x_shift = np.mean(peaks_box[:,0])
    y_shift = np.mean(peaks_box[:,1])

    dist = np.zeros((peaks_box.shape[0],1))
    for p in range(peaks_box.shape[0]):
        dist[p] = np.sqrt( (peaks_box[p,0]-x_shift)**2 + (peaks_box[p,1]-y_shift)**2 )

    peaks_box_cent = peaks_box[np.argmin(dist)]

    peaks_box[:,0] = peaks_box[:,0] - peaks_box_cent[0]
    peaks_box[:,1] = peaks_box[:,1] - peaks_box_cent[1]

    # Apply radial mask
    delta = (1. / pixel_to_angstrom) / 8.
    radius = float(window_size) / 2.
    while peaks_box.shape[0] > reference_lattice.shape[0]:
        del_peaks = np.sqrt(peaks_box[:,0]**2 + peaks_box[:,1]**2) < radius
        peaks_box = peaks_box[del_peaks == True]  
        
        radius = radius - delta

    lattice = peaks_box
    
    return lattice
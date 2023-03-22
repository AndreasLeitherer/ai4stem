import atomap.api as am
import hyperspy.api as hs

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

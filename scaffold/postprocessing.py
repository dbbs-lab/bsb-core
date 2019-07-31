from scipy.stats import truncnorm

def placeParallelFibers(scaffold, granule_geometry, granules):
    parallel_fibers = np.zeros((len(granules_idx),2))
    pf_height = granule_geometry.pf_height
    pf_height_sd = granule_geometry.pf_height_sd
    molecular_layer = scaffold.configuration.getLayer(name='Molecular Layer')
    floor_ml = molecular_layer.Y
    roof_ml = floor_ml + molecular_layer.height # Roof of the molecular layer

    # TODO:
    # Generate a map from granule y position to truncnorm distribution samples so that the re-calculation of the entire distribution per required sample can be avoided.

    for idx, granule in enumerate(granules):
        granule_y = granule[3]
        # Determine min and max height so that the parallel fiber is inside of the molecular layer
        pf_height_min = floor_ml - granule_y
        pf_height_max = roof_ml - granule_y
        # Determine the shape parameters a and b of the truncated normal distribution.
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        a, b = (pf_height_min - pf_height) / pf_height_sd, (pf_height_max - pf_height) / pf_height_sd
        #
    	parallel_fibers[idx,1] = truncnorm.rvs(a, b, size=1) * pf_height_sd + pf_height # Height
        parallel_fibers[idx,0] = granule[0] # ID

    	parallel_fibers[idx,0] = granule[0]
    	parallel_fibers[idx,1] = h_final

    return parallel_fibers

import numpy as np
from astropy.table import Table, join
from astropy import units as u
import h5py
import astropy.coordinates as coord
from astropy.time import Time
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014 as mw
from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
from skycats import funkycat
from copy import deepcopy
from astropy.stats import biweight_midvariance


#-----------------------------------------------------------------------------------------
def loadObsParam(fin, fout):
    '''
    Load APOKASC data

    Parameters
    ----------
    fin : str
        File containing the Kepler, APOGEE and Gaia data
    fout : str
        File containing BASTA results (stellar properties)
  
    Return
    ------
    data : array
        Parameters for the observed stars 
    '''
#-----------------------------------------------------------------------------------------

    # Input data
    kitten = Table.read(fin, format='votable')
    #print (kitten.info)
    kitten = funkycat.standardize(kitten)
    floatfill = funkycat.get_fillvalue(kitten['FE_H_APOGEE'])
    strfill = funkycat.get_fillvalue(kitten['KIC_ID'])
    for col in kitten.columns.keys():
        if '_BASTA' in col:
            kitten.remove_column(col)
    
    # BASTA results
    res_BASTA = Table.read(fout, format='ascii.commented_header', comment='#')
    res_BASTA['starid'] = res_BASTA['starid'].astype('str') 
    res_BASTA.rename_column('starid', 'KIC_ID')
    res_BASTA = funkycat.standardize(res_BASTA, old=np.nan, catid='BASTA')
    #print (res_BASTA.info)
    kitten = join(res_BASTA, kitten, join_type='left', keys='KIC_ID')
    kitten = funkycat.standardize(kitten)
    
    # Select stars with necessary data (ignore stars with incomplete/invalid information)
    # Also check quality of parallaxes
    maskKepler = ((kitten['JMAG_TMASS'] != floatfill) &
                  (kitten['HMAG_TMASS'] != floatfill) &
                  (kitten['KMAG_TMASS'] != floatfill) &
                  (kitten['ERROR_JMAG_TMASS'] != floatfill) &
                  (kitten['ERROR_HMAG_TMASS'] != floatfill) &
                  (kitten['ERROR_KMAG_TMASS'] != floatfill) &
                  (kitten['JMAG_TMASS'] > 0.0) &
                  (kitten['HMAG_TMASS'] > 0.0) &
                  (kitten['KMAG_TMASS'] > 0.0) &
                  (kitten['ERROR_JMAG_TMASS'] > 0.0) &
                  (kitten['ERROR_HMAG_TMASS'] > 0.0) &
                  (kitten['ERROR_KMAG_TMASS'] > 0.0) &
                  (kitten['QFLG_TMASS'] == 'AAA') &
                  (kitten['ME_H_BASTA'] != floatfill) &
                  (kitten['ERROR_ME_H_APOGEE'] != floatfill) &
                  (kitten['TEFF_APOGEE'] != floatfill) &
                  (kitten['ERROR_TEFF_APOGEE'] != floatfill) &
                  (kitten['DNU_SYD_YU'] != floatfill) &
                  (kitten['ERROR_DNU_SYD_YU'] != floatfill) &
                  (kitten['NUMAX_SYD_YU'] != floatfill) &
                  (kitten['ERROR_NUMAX_SYD_YU'] != floatfill) &
                  (kitten['PARALLAX_ZINN_GAIA'] != floatfill) &
                  (kitten['ERROR_PARALLAX_GAIA'] != floatfill) &
                  (kitten['PMRA_GAIA'] != floatfill) &
                  (kitten['ERROR_PMRA_GAIA'] != floatfill) &
                  (kitten['PMDEC_GAIA'] != floatfill) &
                  (kitten['ERROR_PMDEC_GAIA'] != floatfill) &
                  (kitten['RADIAL_VELOCITY_GAIA'] != floatfill) &
                  (kitten['ERROR_RADIAL_VELOCITY_GAIA'] != floatfill) &
                  (kitten['DISTANCE_BASTA'] != floatfill) &
                  (np.isfinite(kitten['ME_H_BASTA']))
                 )

    # Initialize data
    data = np.zeros(len(kitten['PARALLAX_ZINN_GAIA'][maskKepler].data), 
        dtype={'names': ['plx', 'plxErr', 'ra', 'raErr', 'dec', 'decErr', 'rv', 'rvErr', 
        'pm_ra_cosdec', 'pm_ra_cosdecErr', 'pm_dec', 'pm_decErr', 
        'dist', 'distErr', 'age', 'ageErr', 'jmag', 'jmagErr', 'kmag', 'kmagErr', 'teff', 
        'teffErr', 'logg', 'loggErr', 'feh', 'fehErr', 'afe', 'afeErr', 'ofe', 'ofeErr', 
        'mgfe', 'mgfeErr', 'sife', 'sifeErr', 'mass', 'massErr'], 'formats': [float, 
        float, float, float, float, float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float, float, float, float, 
        float, float]})

    data['plx'] = kitten['PARALLAX_ZINN_GAIA'][maskKepler].data           # * u.mas
    data['plxErr'] = kitten['ERROR_PARALLAX_GAIA'][maskKepler].data       # * u.mas
    data['ra'] = kitten['RA_GAIA'][maskKepler].data                       # * u.deg
    data['raErr'] = kitten['ERROR_RA_GAIA'][maskKepler].data * 2.778e-7   # * u.deg
    data['dec'] = kitten['DEC_GAIA'][maskKepler].data                     # * u.deg
    data['decErr'] = kitten['ERROR_DEC_GAIA'][maskKepler].data * 2.778e-7 # * u.deg
    data['rv'] = kitten['RADIAL_VELOCITY_GAIA'][maskKepler].data          # * u.km/u.s
    data['rvErr'] = kitten['ERROR_RADIAL_VELOCITY_GAIA'][maskKepler].data # * u.km/u.s
    data['pm_ra_cosdec'] = kitten['PMRA_GAIA'][maskKepler].data           # * u.mas / u.yr
    data['pm_ra_cosdecErr'] = kitten['ERROR_PMRA_GAIA'][maskKepler].data  # * u.mas / u.yr
    data['pm_dec'] = kitten['PMDEC_GAIA'][maskKepler].data                # * u.mas / u.yr
    data['pm_decErr'] = kitten['ERROR_PMDEC_GAIA'][maskKepler].data       # * u.mas / u.yr
    data['dist'] = 1e-3 * kitten['DISTANCE_BASTA'][maskKepler]            # * u.kpc
    data['distErr'] = 1e-3 * (
        kitten['LOWER_ERROR_DISTANCE_BASTA'][maskKepler] +
        kitten['UPPER_ERROR_DISTANCE_BASTA'][maskKepler]) / 2.            # * u.kpc
    data['age'] = 1e-3 * kitten['AGE_BASTA'][maskKepler]                  # * u.Gyr
    data['ageErr'] = 1e-3 * (
        kitten['LOWER_ERROR_AGE_BASTA'][maskKepler] +
        kitten['UPPER_ERROR_AGE_BASTA'][maskKepler]) / 2.                 # * u.Gyr
    data['jmag'] = kitten['JMAG_TMASS'][maskKepler].data
    data['jmagErr'] = kitten['ERROR_JMAG_TMASS'][maskKepler].data
    data['kmag'] = kitten['KMAG_TMASS'][maskKepler].data
    data['kmagErr'] = kitten['ERROR_KMAG_TMASS'][maskKepler].data
    data['teff'] = kitten['TEFF_APOGEE'][maskKepler].data
    data['teffErr'] = kitten['ERROR_TEFF_APOGEE'][maskKepler].data
    data['logg'] = kitten['LOGG_APOGEE'][maskKepler].data
    data['loggErr'] = kitten['ERROR_LOGG_APOGEE'][maskKepler].data
    data['feh'] = kitten['FE_H_APOGEE'][maskKepler].data
    data['fehErr'] = kitten['ERROR_FE_H_APOGEE'][maskKepler].data
    data['ofe'] = kitten['O_FE_APOGEE'][maskKepler].data
    data['ofeErr'] = kitten['ERROR_O_FE_APOGEE'][maskKepler].data
    data['mgfe'] = kitten['MG_FE_APOGEE'][maskKepler].data
    data['mgfeErr'] = kitten['ERROR_MG_FE_APOGEE'][maskKepler].data
    data['sife'] = kitten['SI_FE_APOGEE'][maskKepler].data
    data['sifeErr'] = kitten['ERROR_SI_FE_APOGEE'][maskKepler].data
    data['afe'] = (data['ofe'] + data['mgfe'] + data['sife']) / 3.
    data['afeErr'] = np.sqrt(data['ofeErr']**2+data['mgfeErr']**2+data['sifeErr']**2)/3.
    data['mass'] = kitten['MASSFIN_BASTA'][maskKepler]
    data['massErr'] = (kitten['LOWER_ERROR_MASSFIN_BASTA'][maskKepler] +
        kitten['UPPER_ERROR_MASSFIN_BASTA'][maskKepler]) / 2. 

    # Ignore stars with unreliable parallaxes
    data = data[data['plx'] > 0.]
    data = data[data['plx'] / data['plxErr'] > 5.]

    return data


#-----------------------------------------------------------------------------------------
def extractKeplerField(mockPath, solarPos='030', starType='bright', 
                       outputFileName='./Kepler_field.hdf5'):
    '''
    Extract simulation stars in the Kepler field-of-view

    Parameters
    ----------
    mockPath : str
        Path for mock catalogue
    solarPos : str
        The solar position assumed in generating the catalogue ('030','120','210','300')
    starType : str 
        Type of star ('bright' or 'faint')
    outputFileName : str
        The name of the output file containing the Kepler field-of-view stars

    Return
    ------
    A file containing the Kepler field-of-view stars    
    '''
#-----------------------------------------------------------------------------------------
    
    # Check the type of stars
    if starType.lower() == 'bright':
        fname1 = 'mock_' + solarPos
    elif starType.lower() == 'faint':
        fname1 = 'mock_faint_' + solarPos
    else:
        raise ValueError('Error: Unrecognized star type!')

    # Kepler field: circle centered on 290.6667, 44.5 (radius 11.2333) 
    # degrees and dec between 44.5-8 and 44.5+8.
    Kepfiera = 290.6667
    Kepfiede = 44.5
    Kepfierad = 11.2333
    Kepfiebor= 8
    
    # Create the file 
    fout = h5py.File(outputFileName,"w")
    fout.close()
    
    # Make the loop
    ifile  = 0
    nfiles = 1
    while ifile < nfiles:
        fname2 = "%s%s.%d.hdf5" % (mockPath, fname1, ifile)
        print("Reading: ", fname2)
        with h5py.File(fname2, "r") as infile:
            if ifile == 0:
                nfiles = infile["Header"].attrs["NumFilesPerSnapshot"]
    
            coords = infile["Stardata/HCoordinates"]
            c = coord.SkyCoord(ra=coords[:, 0] * u.rad, dec=coords[:, 1] * u.rad,
                         frame='icrs')
    
            Kepfield = (c.dec.degree >= Kepfiede - Kepfiebor) & \
                       (c.dec.degree <= Kepfiede + Kepfiebor) & \
                       ((c.ra.degree - Kepfiera) ** 2 + (
                        c.dec.degree - Kepfiede) ** 2 <= Kepfierad ** 2)
    
            if any(Kepfield):
                print('In the Kepler field')
                fout = h5py.File(outputFileName, "r+")
    
                for keyname in infile['Stardata/'].keys():
                    if len(np.shape(infile['Stardata/' + keyname])) > 1:
                        fout['Stardata/'+str(ifile) + '/' + keyname] = \
                            infile['Stardata/' + keyname][Kepfield, :]
                    else:
                        fout['Stardata/'+str(ifile) + '/' + keyname] = \
                            infile['Stardata/' + keyname][Kepfield]
    
                fout.close()
            else:
                print('Not in the field, moving on ...')
    
            infile.close()
            # Next file
            ifile += 1
    return


#-----------------------------------------------------------------------------------------
def loadSimParam(fRaw, fMock, ZXs_Au=0.0181, OFes_Au=4.4340, MgFes_Au=0.5479, 
                 SiFes_Au=0.5144):
    '''
    Load relevant parameters associated with simulation stars in the Kepler field-of-view

    Parameters
    ----------
    fRaw : str
        The name of the file containing simulation snapshot
    fMock : str
        The name of the file containing simulation stars in the Kepler field-of-view
    ZXs_Au : float
        Solar Z/X used in the simulation
    OFes_Au : float
        Solar [O/Fe] used in the simulation
    MgFes_Au : float
        Solar [Mg/Fe] used in the simulation
    SiFes_Au : float
        Solar [Si/Fe] used in the simulation

    Return
    ------
    data : array
        Parameters for the simulation stars
    '''
#-----------------------------------------------------------------------------------------

    # Read raw simulation data
    # This is necessary because certain important quantities, for instance [Fe/H], are not 
    # present in the mock catalogues
    Au = h5py.File(fRaw, 'r')
    stars = Au['/PartType4']
    
    # Process the mock catalogue
    infile = h5py.File(fMock, 'r')
    data = np.zeros(10000000, dtype={'names': ['id', 'mass', 'teff', 'teffErr', 'age', 
        'Z', 'feh', 'afe', 'ofe', 'mgfe', 'mgfe_cor', 'sife', 'ra', 'raErr', 'dec', 'decErr', 
        'plx', 'plxErr', 'pm_ra_cosdec', 'pm_ra_cosdecErr', 'pm_dec', 'pm_decErr', 'rv', 
        'rvErr', 'jmag', 'kmag', 'dist', 'distErr', 'logg', 'loggErr', 'birthZ', 'birthY', 
        'birthX', 'birthVz', 'birthVy', 'birthVx', 'gravPot', 'flag'], 'formats': [int, 
        float, float, float, float, float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float, float, float, float, 
        float, float, float, int]})
    nstar = 0
    partid = -1 
    for keyname in infile['Stardata/'].keys():
        #if (int(keyname) < 86) or (int(keyname) > 88):
        #    continue
    
        print('Processing', keyname)
        for i in range(len(infile['Stardata/' + keyname + '/ParticleID'])):

            # Ignore stars with metallicity below machine precision
            if abs(infile['Stardata/' + keyname + '/Metallicity'][i]) < 1e-14:
                continue

            # Find the index of the star in the raw simulation
            if partid != infile['Stardata/' + keyname + '/ParticleID'][i]:
                partid = infile['Stardata/' + keyname + '/ParticleID'][i]
                indx = np.where(stars['ParticleIDs'][:] == partid)[0][0]

            # Store the data
            data['id'][nstar] = partid 
            data['mass'][nstar] = infile['Stardata/' + keyname + '/Mass'][i]
            data['teff'][nstar] = \
                infile['Stardata/' + keyname + '/EffectiveTemperatureObs'][i]
            data['teffErr'][nstar] = \
                infile['Stardata/' + keyname + '/EffectiveTemperatureError'][i]
            data['logg'][nstar] = \
                infile['Stardata/' + keyname + '/SurfaceGravityObs'][i]
            data['loggErr'][nstar] = \
                infile['Stardata/' + keyname + '/SurfaceGravityError'][i]
            data['age'][nstar] = infile['Stardata/' + keyname + '/Age'][i]
            data['Z'][nstar] = infile['Stardata/' + keyname + '/Metallicity'][i]
            data['feh'][nstar] = (np.log10(data['Z'][nstar] / 
                                  stars['GFM_Metals'][indx, 0]) - np.log10(ZXs_Au))
            tmp = stars['GFM_Metals'][indx, 4] / stars['GFM_Metals'][indx, 8]
            data['ofe'][nstar] = np.log10(tmp) - np.log10(OFes_Au)
            tmp = stars['GFM_Metals'][indx, 6] / stars['GFM_Metals'][indx, 8]
            data['mgfe'][nstar] = np.log10(tmp) - np.log10(MgFes_Au)
            tmp *= 2.
            data['mgfe_cor'][nstar] = np.log10(tmp) - np.log10(MgFes_Au)
            tmp = stars['GFM_Metals'][indx, 7] / stars['GFM_Metals'][indx, 8]
            data['sife'][nstar] = np.log10(tmp) - np.log10(SiFes_Au)
            data['afe'][nstar] = \
                (data['ofe'][nstar] + data['mgfe'][nstar] + data['sife'][nstar]) / 3.
            data['ra'][nstar] = \
                infile['Stardata/' + keyname + '/HCoordinatesObs'][i][0] * 180. / np.pi
            data['raErr'][nstar] = \
                infile['Stardata/' + keyname + '/HCoordinateErrors'][i][0] * 180. / np.pi
            data['dec'][nstar] = \
                infile['Stardata/' + keyname + '/HCoordinatesObs'][i][1] * 180. / np.pi
            data['decErr'][nstar] = \
                infile['Stardata/' + keyname + '/HCoordinateErrors'][i][1] * 180. / np.pi
            data['plx'][nstar] = \
                infile['Stardata/' + keyname + '/HCoordinatesObs'][i][2] * 1e3
            data['plxErr'][nstar] = \
                infile['Stardata/' + keyname + '/HCoordinateErrors'][i][2] * 1e3
            data['dist'][nstar] = 1. / data['plx'][nstar]
            data['distErr'][nstar] = data['plxErr'][nstar] / data['plx'][nstar]**2
            data['pm_ra_cosdec'][nstar] = \
                infile['Stardata/' + keyname + '/HVelocitiesObs'][i][0] * 1e3
            data['pm_ra_cosdecErr'][nstar] = \
                infile['Stardata/' + keyname + '/HVelocityErrors'][i][0] * 1e3
            data['pm_dec'][nstar] = \
                infile['Stardata/' + keyname + '/HVelocitiesObs'][i][1] * 1e3
            data['pm_decErr'][nstar] = \
                infile['Stardata/' + keyname + '/HVelocityErrors'][i][1] * 1e3
            data['rv'][nstar] = \
                infile['Stardata/' + keyname + '/HVelocitiesObs'][i][2]
            data['rvErr'][nstar] = \
                infile['Stardata/' + keyname + '/HVelocityErrors'][i][2]
            data['jmag'][nstar] = infile['Stardata/' + keyname + '/Magnitudes'][i][3]
            data['kmag'][nstar] = infile['Stardata/' + keyname + '/Magnitudes'][i][5]
            data['birthZ'][nstar] = stars['BirthPos'][indx, 0] * 1e3
            data['birthY'][nstar] = stars['BirthPos'][indx, 1] * 1e3
            data['birthX'][nstar] = stars['BirthPos'][indx, 2] * 1e3
            data['birthVz'][nstar] = stars['BirthVel'][indx, 0]
            data['birthVy'][nstar] = stars['BirthVel'][indx, 1]
            data['birthVx'][nstar] = stars['BirthVel'][indx, 2]
            data['gravPot'][nstar] = stars['GravPotential'][indx]
            data['flag'][nstar] = infile['Stardata/' + keyname + '/AccretedFlag'][i]
            nstar = nstar + 1
    infile.close()
    Au.close()
    data = data[0:nstar]
    
    # Short stars according to their SSP particle id 
    data = data[data['id'].argsort()]

    # Ignore stars with unreliable parallaxes
    data = data[data['plx'] > 0.]
    data = data[data['plx'] / data['plxErr'] > 5.]

    return data


#-----------------------------------------------------------------------------------------
def applySelectionFunc(obsStarsData, simStarsData, njk=20, nk=20, npi=20, jkmin=0., 
                       jkmax=20., kmin=0., kmax=20, pimin=0., pimax=2.):
    '''
    Apply the selection function

    Parameters
    ----------
    obsStarsData : array
        Observed data 
    simStarsData : array
        Simulation data
    njk : int
        Number of points along J - K dimension
    nk : int
        Number of points along K dimension
    npi : int
        Number of points along parallax dimension
    jkmin : float
        Minimum value of J - K for the observed sample
    jkmax : float
        Maximum value of J - K for the observed sample
    kmin : float
        Minimum value of K for the observed sample
    kmax : float
        Maximum value of K for the observed sample
    pimin : float
        Minimum value of parallax for the observed sample
    pimax : float
        Maximum value of parallax for the observed sample

    Return
    ------
    obsStars : array
        Observed data after selection function
    simStars : array
        simulation data after selection function
    '''
#-----------------------------------------------------------------------------------------

    # Remove simulation stars that are outside the observed J - K, K and pi ranges
    simStarsData = simStarsData[simStarsData['jmag'] - simStarsData['kmag'] >= jkmin]
    simStarsData = simStarsData[simStarsData['jmag'] - simStarsData['kmag'] <= jkmax]
    simStarsData = simStarsData[simStarsData['kmag'] >= kmin]
    simStarsData = simStarsData[simStarsData['kmag'] <= kmax]
    simStarsData = simStarsData[simStarsData['plx'] >= pimin]
    simStarsData = simStarsData[simStarsData['plx'] <= pimax]

    # Break the data into small cells and store stars within them in a list
    # K, J - K, pi => x, y, z
    jkstep = (jkmax - jkmin) / njk
    kstep  = (kmax - kmin) / nk
    pistep = (pimax - pimin) / npi
    obsList = []
    simList = []
    for i in range(nk):
        xko = obsStarsData[obsStarsData['kmag'] >= kmin + i * kstep]
        xko = xko[xko['kmag'] < kmin + (i+1) * kstep]
        xks = simStarsData[simStarsData['kmag'] >= kmin + i * kstep]
        xks = xks[xks['kmag'] < kmin + (i+1) * kstep]
        for j in range(njk):
            xjko = xko[xko['jmag'] - xko['kmag'] >= jkmin + j * jkstep]
            xjko = xjko[xjko['jmag'] - xjko['kmag'] < jkmin + (j+1) * jkstep]
            xjks = xks[xks['jmag'] - xks['kmag'] >= jkmin + j * jkstep]
            xjks = xjks[xjks['jmag'] - xjks['kmag'] < jkmin + (j+1) * jkstep]
            for k in range(npi):
                xpo = xjko[xjko['plx'] >= pimin + k * pistep]
                xpo = xpo[xpo['plx'] < pimin + (k+1) * pistep]
                xps = xjks[xjks['plx'] >= pimin + k * pistep]
                xps = xps[xps['plx'] < pimin + (k+1) * pistep]
                if len(xpo) > 0 and len(xps) > 0:
                    obsList.append(xpo)
                    simList.append(xps)
    
    # Pick up same number of stars for the simulation in each cell 
    simStars = deepcopy(simStarsData[:])    
    obsStars = deepcopy(obsStarsData[:])
    nstars = 0
    for i in range(len(obsList)):
        n1 = len(obsList[i])
        n2 = len(simList[i])
        np.random.seed(30)
        if n1 >= n2:
            np.random.shuffle(obsList[i])   
            obsStars[nstars:nstars+n2] = obsList[i][:n2]
            simStars[nstars:nstars+n2] = simList[i][:]
            nstars += n2
        else:
            np.random.shuffle(simList[i])   
            simStars[nstars:nstars+n1] = simList[i][:n1]
            obsStars[nstars:nstars+n1] = obsList[i][:]
            nstars += n1
    obsStars = obsStars[0:nstars]
    simStars = simStars[0:nstars]

    return obsStars, simStars


#-----------------------------------------------------------------------------------------
def cartesianToCylindrical(pos, vel, R0=8., Z0=20., V0=240., Vlsd=[11.1, 12.24, 7.25]):
    '''
    Transform position and velocity from cartesian to cylindrical coordinate

    Parameters
    ----------
    pos : tuple
        Position in cartesian coordinate (in kpc)
    vel : tuple
        Velocity in cartesian coordinate (in km/s)
    R0 : float
        Radial distance of the Sun from the Galactic center (in kpc)
    Z0 : float
        Height of the Sun above the Galactic plane (in pc)
    V0 : float
        Rotational velocity of the Sun (in km/s)
    Vlsd : list
        Velocity of the Sun w.r.t the local standard of rest (in km/s)

    Return
    ------
    position : array
        Position in galactocentric cylindrical coordinate 
    velocity : array
        Velocity in galactocentric cylindrical coordinate
    '''
#-----------------------------------------------------------------------------------------

    x, y, z = pos[0] * u.kpc, pos[1] * u.kpc, pos[2] * u.kpc
    v_x, v_y, v_z = vel[0] * u.km / u.s, vel[1] * u.km / u.s, vel[2] * u.km / u.s 
    v_sun = coord.CartesianDifferential([Vlsd[0], V0 + Vlsd[1], Vlsd[2]] * u.km / u.s)
    gc = coord.Galactocentric(x=x, y=y, z=z, v_x=v_x, v_y=v_y, v_z=v_z,
             representation_type='cartesian', differential_type='cartesian',
             galcen_distance=R0 * u.kpc, galcen_v_sun=v_sun, z_sun=Z0 * u.pc)
    gc.representation_type = 'cylindrical'
    gc.differential_type = 'cylindrical'
    
    position = np.zeros(len(x),
                 dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
    velocity = np.zeros(len(x),
                 dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
    position['rho'] = gc.rho.to(u.kpc).value
    position['phi'] = gc.phi.to(u.rad).value
    position['z'] = gc.z.to(u.kpc).value
    velocity['rho'] = gc.d_rho.to(u.km / u.s).value
    velocity['phi'] = (gc.d_phi.to(u.mas / u.yr).value * 4.74047 * gc.rho.to(u.kpc).value *
                       u.km / u.s).value
    velocity['z'] = gc.d_z.to(u.km / u.s).value

    return position, velocity


#-----------------------------------------------------------------------------------------
def computeKinematics(ra, dec, dist, pm_ra, pm_dec, rv, R0=8., Z0=20., V0=240., 
        Vlsd=[11.1, 12.24, 7.25]):
    '''
    Compute kinematics

    Parameters
    ----------
    ra : float
        Right ascension of the star (in degree)
    dec : float
        Declination of the star (in degree)
    dist : float
        Distance to the star (in kpc)
    pm_ra : float
        Proper motion in right ascension * cos(dec) (in mas/yr)
    pm_dec : float
        Proper motion in declination (in mas/yr)
    rv : float
        Radial velocity (km/s)
    R0 : float
        Radial distance of the Sun from the Galactic center (in kpc)
    Z0 : float
        Height of the Sun above the Galactic plane (in pc)
    V0 : float
        Rotational velocity of the Sun (in km/s)
    Vlsd : list
        Velocity of the Sun w.r.t the local standard of rest (in km/s)

    Return
    ------
    pos : array
        Position in galactocentric cylindrical coordinate 
    vel : array
        Velocity in galactocentric cylindrical coordinate
    act : array
        Action in galactocentric cylindrical coordinate
    orb : array
        Orbital parameters (maximum height and eccentricity) 
    '''
#-----------------------------------------------------------------------------------------

    # Tranform to galactocentric cylindrical coordinate
    v_sun = coord.CartesianDifferential([Vlsd[0], V0 + Vlsd[1], Vlsd[2]] * u.km / u.s)
    cs = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=dist * u.kpc,
                        pm_ra_cosdec=pm_ra * u.mas / u.yr, pm_dec=pm_dec * u.mas / u.yr, 
                        radial_velocity=rv * u.km / u.s, galcen_distance=R0 * u.kpc, 
                        galcen_v_sun=v_sun, z_sun=Z0 * u.pc
                       )
    gc = coord.Galactocentric(galcen_distance=R0 * u.kpc, galcen_v_sun=v_sun, 
                              z_sun=Z0 * u.pc)
    cs = cs.transform_to(gc)
    cs.representation_type = 'cylindrical'
    cs.differential_type = 'cylindrical'
    
    # Position in cylindrical coordinate
    rho, phi, z = cs.rho.to(u.kpc), cs.phi.to(u.rad), cs.z.to(u.kpc)
    pos = np.zeros(1,
              dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]}) 
    pos['rho'][0], pos['phi'][0], pos['z'][0] = rho.value, phi.value, z.value 

    # Velocity in cylindrical coordinate
    vrho = cs.d_rho.to(u.km / u.s)
    vphi = (cs.d_phi.to(u.mas / u.yr).value * 4.74047 * cs.rho.to(u.kpc).value * 
            u.km / u.s)
    vz = cs.d_z.to(u.km / u.s)
    vel = np.zeros(1,
              dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]}) 
    vel['rho'][0], vel['phi'][0], vel['z'][0] = vrho.value, vphi.value, vz.value

    # Compute actions
    delta = estimateDeltaStaeckel(mw, rho, z)
    aAS = actionAngleStaeckel(pot=mw, delta=delta, c=True, ro=R0*u.kpc, vo=V0*u.km / u.s)
    jrho, jphi, jz = aAS(rho, vrho, vphi, z, vz, fixed_quad=True)
    act = np.zeros(1,
              dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]}) 
    act['rho'][0], act['phi'][0], act['z'][0] = jrho.value, jphi.value, jz.value

    # Compute maximum height and eccentricity
    o = Orbit([rho, vrho, vphi, z, vz], ro=R0 * u.kpc, vo=V0 * u.km / u.s)
    np.seterr(over='raise')
    zmax, ecc = -1., -1.
    try:
        zmax = o.zmax(pot=mw, delta=delta, analytic=True, type='staeckel')
    except FloatingPointError:
        pass
    try:
        ecc = o.e(pot=mw, delta=delta, analytic=True, type='staeckel')
    except FloatingPointError:
        pass
    orb = np.zeros(1, dtype={'names': ['zmax', 'ecc'], 'formats': [float, float]}) 
    orb['zmax'][0], orb['ecc'][0] = zmax.value, ecc

    return pos, vel, act, orb


#-----------------------------------------------------------------------------------------
def dispersionYasX(x, y, nbins=10):
    '''
    Dispersion of Y as a function of X

    Parameters
    ----------
    x : array
        Independent variable
    y : array
        Dependent variable
    nbins : int
        Number of bins

    Return
    ------
    dx : array
        Binned x
    sigmay : array
        Dispersion of y
    '''

    dx = np.zeros(nbins)
    sigmay = np.zeros(nbins)

    xstep = (np.amax(x) - np.amin(x)) / nbins
    xbin = np.amin(x)
    for i in range(nbins):
        mask = np.logical_and(x >= xbin, x < xbin + xstep)
        ybin = y[mask]
        sigmay[i] = np.sqrt(biweight_midvariance(ybin))
        #sigmay[i] = np.std(ybin)
        dx[i] = xbin + 0.5 * xstep
        xbin += xstep

    return dx, sigmay 


#-----------------------------------------------------------------------------------------
def majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, nxminor=5, nymajor=7, nyminor=5):
    '''
    Calculate step sizes for major and minor tick levels

    Parameters
    ----------
    xmin : float
        Minimum value of x
    xmax : float
        Maximum value of x
    ymin : float
        Minimum value of y
    ymax : float 
        Maximum value of y
    nxmajor : int 
        Typical number of required major ticks on the x-axis
    nxminor : int 
        Number of required minor ticks between two consecutive major ticks on the x-axis
    nymajor : int 
        Typical number of required major ticks on the y-axis
    nyminor : int 
        Number of required minor ticks between two consecutive major ticks on the y-axis

    Return
    ------
    xmajor : float
        Step size for major ticks on the x-axis
    xminor : float
        Step size for minor ticks on the x-axis
    ymajor : float
        Step size for major ticks on the y-axis
    yminor : float
        Step size for minor ticks on the y-axis
    '''
#-----------------------------------------------------------------------------------------

    xmajor = float("{:.0e}".format((xmax - xmin) / nxmajor))
    xminor = xmajor / nxminor
    ymajor = float("{:.0e}".format((ymax - ymin) / nymajor))
    yminor = ymajor / nyminor
    
    return xmajor, xminor, ymajor, yminor


#-----------------------------------------------------------------------------------------
def binMeanStd(x, y, nbins=10):
    '''
    Compute binned median and standard deviation 

    Parameters
    ----------
    x : array
        Independent variable
    y : array
        Dependent variable
    nbins : int
        Number of bins

    Return
    ------
    data : array
        Binned medean and standard deviations for both x and y
    '''
#-----------------------------------------------------------------------------------------

    # Sort x and y according to x
    indc = np.argsort(x)
    xsort, ysort = x[indc], y[indc]

    # Compute mean and standard deviation for each bin
    nstars = len(xsort)
    nstars_bin = int(nstars / nbins + 1.)
    data = np.zeros((nbins, 4))
    for i in range(nbins-1):
        data[i, 0] = np.median(xsort[i * nstars_bin:(i + 1) * nstars_bin])
        data[i, 1] = np.median(ysort[i * nstars_bin:(i + 1) * nstars_bin])
        data[i, 2] = np.sqrt(biweight_midvariance(xsort[i * nstars_bin:(i + 1) * nstars_bin]))
        data[i, 3] = np.sqrt(biweight_midvariance(ysort[i * nstars_bin:(i + 1) * nstars_bin]))
    data[nbins-1, 0] = np.median(xsort[(nbins - 1) * nstars_bin:nstars])
    data[nbins-1, 1] = np.median(ysort[(nbins - 1) * nstars_bin:nstars])
    data[nbins-1, 2] = np.sqrt(biweight_midvariance(xsort[(nbins - 1) * nstars_bin:nstars]))
    data[nbins-1, 3] = np.sqrt(biweight_midvariance(ysort[(nbins - 1) * nstars_bin:nstars]))

    return data


#-----------------------------------------------------------------------------------------
def joinSnap2(path):
    '''
    Join Auriga level 2 snapshots 

    Parameters
    ----------
    path : str
        Path to the snapshots

    Return
    ------
    A file containing joined snapshots
    '''
#-----------------------------------------------------------------------------------------

    # Read first snapshot
    Au = h5py.File(path + 'snapshot_highfreqstars_804.0.hdf5', 'r')
    stars = Au['/PartType4']

    # Total number of particles
    nt = Au['Header'].attrs['NumPart_Total'][4]

    # Write to the hdf file
    f = h5py.File(path + 'join.hdf5', 'w')
    pid = f.create_dataset('/PartType4/ParticleIDs', (nt, ), dtype='<u8')
    gfm = f.create_dataset('/PartType4/GFM_Metals', (nt, 9), dtype='<f4')
    pos = f.create_dataset('/PartType4/BirthPos', (nt, 3), dtype='<f4')
    vel = f.create_dataset('/PartType4/BirthVel', (nt, 3), dtype='<f4')
    pot = f.create_dataset('/PartType4/GravPotential', (nt, ), dtype='<f4')

    # Write to the arrays
    n1 = 0
    n2 = Au['Header'].attrs['NumPart_ThisFile'][4]
    for i in range(32):
        print (i, n2, nt)
        pid[n1:n2] = stars['ParticleIDs'][:]
        gfm[n1:n2, :] = stars['GFM_Metals'][:, :]
        pos[n1:n2, :] = stars['BirthPos'][:, :]
        vel[n1:n2, :] = stars['BirthVel'][:, :]
        pot[n1:n2] = stars['Potential'][:]
        if i+1 <= 31:
            Au = h5py.File(path + 'snapshot_highfreqstars_804.' + str(i+1) + '.hdf5', 'r')
            stars = Au['/PartType4']
            n1 = n2
            n2 = n1 + Au['Header'].attrs['NumPart_ThisFile'][4]

    return


# The following functions were used for certain testing purposes
#---------------------------------------------------------------
#
###########################################################################################
## Apply selection function on simulation stars 
###########################################################################################
#def applySelectionFunc(obsStarsData, simStarsData, njk=20, nk=20, npi=20, nhi=20, 
#                       jkmin=None, jkmax=None, kmin=None, kmax=None, pimin=None, 
#                       pimax=None, himin=None, himax=None):
#
#    # Calculate the observed ranges in J - K, K and pi (if not given)
#    if jkmin is None:
#        jkmin = np.amin(obsStarsData['jmag'] - obsStarsData['kmag'])
#    if jkmax is None:
#       jkmax = np.amax(obsStarsData['jmag'] - obsStarsData['kmag'])
#    if kmin is None:
#       kmin = np.amin(obsStarsData['kmag'])
#    if kmax is None:
#       kmax = np.amax(obsStarsData['kmag'])
#    if pimin is None:
#       pimin = np.amin(obsStarsData['plx'])
#    if pimax is None:
#       pimax = np.amax(obsStarsData['plx'])
#
#    # Remove simulation stars that are outside the observed J - K, K and pi ranges
#    simStarsData = simStarsData[simStarsData['jmag'] - simStarsData['kmag'] >= jkmin]
#    simStarsData = simStarsData[simStarsData['jmag'] - simStarsData['kmag'] <= jkmax]
#    simStarsData = simStarsData[simStarsData['kmag'] >= kmin]
#    simStarsData = simStarsData[simStarsData['kmag'] <= kmax]
#    simStarsData = simStarsData[simStarsData['plx'] >= pimin]
#    simStarsData = simStarsData[simStarsData['plx'] <= pimax]
#
#    # Calculate vertical height of stars
#    R0 = 8. * u.kpc      # Reid et al 2014
#    V0 = 240 * u.km / u.s  # Reid et al 2014
#    z_sun = 20 * u.pc      # Chen et al 2001
#    v_sun = coord.CartesianDifferential([11.1, 240 + 12.24, 7.25] * u.km / u.s)
#    gc = coord.Galactocentric(galcen_distance=R0, galcen_v_sun=v_sun, z_sun=z_sun)
#
#    csobs = coord.SkyCoord(ra=obsStarsData['ra'] * u.deg, dec=obsStarsData['dec'] * u.deg, 
#                distance=obsStarsData['dist'] * u.kpc, galcen_distance=R0, 
#                galcen_v_sun=v_sun, z_sun=z_sun)
#    csobs = csobs.transform_to(gc)
#    csobs.representation_type = 'cylindrical'
#    zobs = csobs.z.to(u.kpc)
#    if himin is None:
#       himin = np.amin(zobs)
#    if himax is None:
#       himax = np.amax(zobs)
#    
#    cssim = coord.SkyCoord(ra=simStarsData['ra'] * u.deg, 
#                dec=simStarsData['dec'] * u.deg, distance=simStarsData['dist'] * u.kpc,
#                galcen_distance=R0, galcen_v_sun=v_sun, z_sun=z_sun)
#    cssim = cssim.transform_to(gc)
#    cssim.representation_type = 'cylindrical'
#    zsim = cssim.z.to(u.kpc)
#
#    # Break the data into small cells and store stars within them in a list
#    # K, J - K, pi => x, y, z
#    jkstep = (jkmax - jkmin) / njk
#    kstep  = (kmax - kmin) / nk
#    pistep = (pimax - pimin) / npi
#    histep = (himax - himin) / nhi
#    obsList = []
#    simList = []
#    for i in range(nk):
#        xko = obsStarsData[obsStarsData['kmag'] >= kmin + i * kstep]
#        zko = zobs[obsStarsData['kmag'] >= kmin + i * kstep]
#        zko = zko[xko['kmag'] < kmin + (i+1) * kstep]
#        xko = xko[xko['kmag'] < kmin + (i+1) * kstep]
#        xks = simStarsData[simStarsData['kmag'] >= kmin + i * kstep]
#        zks = zsim[simStarsData['kmag'] >= kmin + i * kstep]
#        zks = zks[xks['kmag'] < kmin + (i+1) * kstep]
#        xks = xks[xks['kmag'] < kmin + (i+1) * kstep]
#        for j in range(njk):
#            xjko = xko[xko['jmag'] - xko['kmag'] >= jkmin + j * jkstep]
#            zjko = zko[xko['jmag'] - xko['kmag'] >= jkmin + j * jkstep]
#            zjko = zjko[xjko['jmag'] - xjko['kmag'] < jkmin + (j+1) * jkstep]
#            xjko = xjko[xjko['jmag'] - xjko['kmag'] < jkmin + (j+1) * jkstep]
#            xjks = xks[xks['jmag'] - xks['kmag'] >= jkmin + j * jkstep]
#            zjks = zks[xks['jmag'] - xks['kmag'] >= jkmin + j * jkstep]
#            zjks = zjks[xjks['jmag'] - xjks['kmag'] < jkmin + (j+1) * jkstep]
#            xjks = xjks[xjks['jmag'] - xjks['kmag'] < jkmin + (j+1) * jkstep]
#            for k in range(npi):
#                xpo = xjko[xjko['plx'] >= pimin + k * pistep]
#                zpo = zjko[xjko['plx'] >= pimin + k * pistep]
#                zpo = zpo[xpo['plx'] < pimin + (k+1) * pistep]
#                xpo = xpo[xpo['plx'] < pimin + (k+1) * pistep]
#                xps = xjks[xjks['plx'] >= pimin + k * pistep]
#                zps = zjks[xjks['plx'] >= pimin + k * pistep]
#                zps = zps[xps['plx'] < pimin + (k+1) * pistep]
#                xps = xps[xps['plx'] < pimin + (k+1) * pistep]
#                for l in range(nhi):
#                    xho = xpo[zpo >= himin + l * histep]
#                    zho = zpo[zpo >= himin + l * histep]
#                    xho = xho[zho < himin + (l+1) * histep]
#                    xhs = xps[zps >= himin + l * histep]
#                    zhs = zps[zps >= himin + l * histep]
#                    xhs = xhs[zhs < himin + (l+1) * histep]
#                    if len(xho) > 0 and len(xhs) > 0:
#                        obsList.append(xho)
#                        simList.append(xhs)
#    
#    # Pick up same number of stars for the simulation in each cell 
#    simStars = simStarsData[:]    
#    obsStars = obsStarsData[:]
#    nstars = 0
#    for i in range(len(obsList)):
#        n1 = len(obsList[i])
#        n2 = len(simList[i])
#        if n1 >= n2:
#            np.random.shuffle(obsList[i])   
#            obsStars[nstars:nstars+n2] = obsList[i][:n2]
#            simStars[nstars:nstars+n2] = simList[i][:]
#            nstars += n2
#        else:
#            np.random.shuffle(simList[i])   
#            simStars[nstars:nstars+n1] = simList[i][:n1]
#            obsStars[nstars:nstars+n1] = obsList[i][:]
#            nstars += n1
#    obsStars = obsStars[0:nstars]
#    simStars = simStars[0:nstars]
#
#    return obsStars, simStars

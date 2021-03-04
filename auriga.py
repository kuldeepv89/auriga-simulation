import os
import sys
import numpy as np
from astropy import units as u
import astropy.coordinates as coord
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter, LogLocator
from matplotlib.patches import Rectangle
import seaborn as sns
from galpy.actionAngle import UnboundError
import pickle
from scipy.stats import rv_histogram, ks_2samp, anderson_ksamp
import supportAuriga as sa

# Don't print Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', UserWarning)



# Halo number, catalogue ('HITS' or 'ICC'), solar position ('030', '120', '210' or '300')
#-----------------------------------------------------------------------------------------
AuHaloNum, catalogue, solarPos = 6, 'ICC', '030'
AuHaloName = 'Au' + str(AuHaloNum).zfill(2)



# PATHS for various I/O folders
#-----------------------------------------------------------------------------------------
# Path for APOKASC data
apokascPath = 'data/apokasc2/'

# Path for the mock catalogues 
if catalogue.lower() == 'hits':
    mockpath = 'data/' + AuHaloName + '_HITS/v2/kroupaIMF/level3_MHD/halo_' \
               + str(AuHaloNum) + '/mockdir_angle' + solarPos + '/'
elif catalogue.lower() == 'icc':
    mockpath = 'data/' + AuHaloName + '_ICC/v3/chabrierIMF/level3_MHD/halo_' \
               + str(AuHaloNum) + '/mockdir_angle' + solarPos + '/'
else:
    print ('Catalogue is not recognized. Terminating the run...')
    sys.exit(1)

# Path for the simulation snapshot
snappath = 'data/snapshot_reduced_halo_' + str(AuHaloNum) + '_063.hdf5'

# Path for the output data
plotpath = 'data/outputs/' 



# The code should run without human intervention from here onward
# Extract parameters for the observed stars
#-----------------------------------------------------------------------------------------
print ()
print ('Loading observed stars...')

obsStars = sa.loadObsParam(apokascPath + 'Apokasc_SER_NUM_TMASS.vot', 
    apokascPath + 'Kepler_APOGEE_BASTA_vers0.32_Dec2020_SER_NUM_TMASS.ascii')

print ('Total number of observed stars = %d' %(len(obsStars)))



# Extract parameters for simulation stars in the Kepler field-of-view
#-----------------------------------------------------------------------------------------
print ()
print ('Loading simulation stars...')

# If not already done, extract the Kepler field-of-view in a file 
fname = plotpath + 'Kepler_' + AuHaloName + '.hdf5'
if not os.path.exists(fname): 
    _ = sa.extractKeplerField(mockpath, solarPos=solarPos, starType='bright', 
                           outputFileName=fname)

if catalogue.lower() == 'hits': # Also extract faint stars
    fname = plotpath + 'Kepler_faint_' + AuHaloName + '.hdf5'
    if not os.path.exists(fname): 
        _ = sa.extractKeplerField(mockpath, solarPos=solarPos, starType='faint', 
                               outputFileName=fname)    

# If not already done, extract parameters in a file
fname = plotpath + 'Kepler_' + AuHaloName + '.pkl'
if not os.path.exists(fname):
    fMock = plotpath + 'Kepler_' + AuHaloName + '.hdf5'
    brightStarsData = sa.loadSimParam(snappath, fMock, ZXs_Au = 0.0181, OFes_Au=4.4340, 
                                     MgFes_Au=0.5479, SiFes_Au=0.5144)
    with open(fname, 'wb') as ff:
        pickle.dump(brightStarsData, ff)

if catalogue.lower() == 'hits': # Also extract for faint stars
    fname = plotpath + 'Kepler_faint_' + AuHaloName + '.pkl'
    if not os.path.exists(fname): 
        fMock = plotpath + 'Kepler_faint_' + AuHaloName + '.hdf5'
        faintStarsData = sa.loadSimParam(snappath, fMock, ZXs_Au = 0.0181, OFes_Au=4.4340, 
                                        MgFes_Au=0.5479, SiFes_Au=0.5144)
        with open(fname, 'wb') as ff:
            pickle.dump(faintStarsData, ff)


# Load parameters
with open(plotpath + 'Kepler_' + AuHaloName + '.pkl', 'rb') as ff:
    simStars = pickle.load(ff)

if catalogue.lower() == 'hits': # Also load for faint stars
    with open(plotpath + 'Kepler_faint_' + AuHaloName + '.pkl', 'rb') as ff:
        faintStarsData = pickle.load(ff)
        simStars = np.hstack((simStars, faintStarsData))

print ('Total number of simulation stars = %d' %(len(simStars)))



# Apply selection function
#-----------------------------------------------------------------------------------------
# Calculate the observed ranges in J - K, K and pi 
jkmin = np.amin(obsStars['jmag'] - obsStars['kmag'])
jkmax = np.amax(obsStars['jmag'] - obsStars['kmag'])
kmin  = np.amin(obsStars['kmag'])
kmax  = np.amax(obsStars['kmag'])
pimin = np.amin(obsStars['plx'])
pimax = np.amax(obsStars['plx'])

njk, nk, npi = 5, 5, 5
for i in range(20):

    print ()
    print ('njk, nk, npi = %d, %d, %d' %(njk, nk, npi))
 
    # Selection function
    obsStarsData, simStarsData = sa.applySelectionFunc(obsStars, simStars, njk=njk, nk=nk, 
        npi=npi, jkmin=jkmin, jkmax=jkmax, kmin=kmin, kmax=kmax, pimin=pimin, pimax=pimax)

    # Test J - K distribution
    obsJK = obsStarsData['jmag'] - obsStarsData['kmag']
    simJK = simStarsData['jmag'] - simStarsData['kmag']
    statjk, critical_values, pvalue = anderson_ksamp([obsJK, simJK])
    if (statjk >= critical_values[2]):
        njk += 5

    # Test K distribution
    statk, critical_values, pvalue = anderson_ksamp([obsStarsData['kmag'], 
                                                    simStarsData['kmag']])
    if (statk >= critical_values[2]):
        nk += 5

    # Test parallax distribution
    statpi, critical_values, pvalue = anderson_ksamp([obsStarsData['plx'], 
                                                     simStarsData['plx']])
    if (statpi >= critical_values[2]):
        npi += 5

    # Print test statistics and check if they are within the critical value
    print ('Test statistics for J - K, K and parallax and critical value at 5%% level = '
           '%.4f, %.4f, %.4f, %.4f' 
           %(statjk, statk, statpi, critical_values[2]))
    if ((statjk < critical_values[2]) and (statk < critical_values[2]) and 
        (statpi < critical_values[2])):
        break

# Check
if i == 19:
    print ('Too small cell size: selection function failed. Terminating the run...')
    sys.exit(2)

# Print 
print ()
print ('Final numbers of observed, simulation stars = %d, %d' %(len(obsStarsData), 
       len(simStarsData)))
print ('Number of in-situ, accreted, sub-halo stars = %d, %d, %d' 
       %(len(simStarsData['age'][simStarsData['flag']==-1]), 
       len(simStarsData['age'][simStarsData['flag']==0]), 
       len(simStarsData['age'][simStarsData['flag']==1])))



# Calculate Kinematics of stars 
#-----------------------------------------------------------------------------------------
# Observed stars
print ()
print ('Computing actions for observed stars...')
n = len(obsStarsData['ra'])
obsPos = np.zeros(n, 
             dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
obsVel = np.zeros(n, 
             dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
obsAct = np.zeros(n, 
             dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
obsOrb = np.zeros(n, dtype={'names': ['zmax', 'ecc'], 'formats': [float, float]})
for i in range(n):
    try:
        obsPos[i], obsVel[i], obsAct[i], obsOrb[i] = sa.computeKinematics(
            obsStarsData['ra'][i], obsStarsData['dec'][i], obsStarsData['dist'][i], 
            obsStarsData['pm_ra_cosdec'][i], obsStarsData['pm_dec'][i], 
            obsStarsData['rv'][i], R0=8., Z0=20., V0=240., Vlsd=[11.1, 12.24, 7.25])
    except UnboundError:
        print ('Warning: Orbit seems to be unbound')

# Simulation stars
print ('Computing actions for simulation stars...')
simPos = np.zeros(n, 
             dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
simVel = np.zeros(n, 
             dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
simAct = np.zeros(n, 
             dtype={'names': ['rho', 'phi', 'z'], 'formats': [float, float, float]})
simOrb = np.zeros(n, dtype={'names': ['zmax', 'ecc'], 'formats': [float, float]})
for i in range(n):
    try:
        simPos[i], simVel[i], simAct[i], simOrb[i] = sa.computeKinematics(
            simStarsData['ra'][i], simStarsData['dec'][i], simStarsData['dist'][i], 
            simStarsData['pm_ra_cosdec'][i], simStarsData['pm_dec'][i], 
            simStarsData['rv'][i], R0=8., Z0=20., V0=240., Vlsd=[11.1, 12.24, 7.25])
    except UnboundError:
        print ('Warning: Orbit seems to be unbound')



# Transform birth positions and velocities to cylindrical coordinate
#-----------------------------------------------------------------------------------------
birthPos, birthVel = sa.cartesianToCylindrical(
    (simStarsData['birthX'], simStarsData['birthY'], simStarsData['birthZ']), 
    (simStarsData['birthVx'], simStarsData['birthVy'], simStarsData['birthVz']), 
    R0=8., Z0=20., Vlsd=[11.1, 12.24, 7.25])



# Fix font sizes for plots according to the number of panels
#-----------------------------------------------------------------------------------------
print ()
print ('Generating plots...')
lab1  = 22
tick1 = 18
lpad1 = 3
tpad1 = 3
legs1 = 18

lab12  = 14
tick12 = 11
lpad12 = 2
tpad12 = 2
legs12 = 11

lab13  = 12
tick13 = 9
lpad13 = 1
tpad13 = 1
legs13 = 9

color = ['#D55E00', '#56B4E9', '#000000', 'darkgrey', '#6C9D34', '#482F76', '#AE7839', 
         '#C44E52', '#4C72B0']



# Generate metallicity dispersion vs. age plot 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14., 0., 0.5 

# Plot observed stars 
data = sa.binMeanStd(obsStarsData['age'], obsStarsData['feh'], nbins=10)
ax1.plot(data[:, 0], data[:, 3], '-', lw=2, rasterized=True, color=color[0])
ax1.plot(data[:, 0], data[:, 3], 'o', ms=3, rasterized=True, c=color[0], 
         label=r'Observed')
#x, y = sa.dispersionYasX(obsStarsData['age'], obsStarsData['feh'], nbins=10)
#ax1.plot(x, y, ':', lw=2, rasterized=True, color=color[0])
#ax1.plot(x, y, 'o', ms=3, rasterized=True, c=color[0])

# Plot simulated stars 
data = sa.binMeanStd(simStarsData['age'], simStarsData['feh'], nbins=10)
ax1.plot(data[:, 0], data[:, 3], '--', lw=2, rasterized=True, color=color[1])
ax1.plot(data[:, 0], data[:, 3], 'o', ms=3, rasterized=True, c=color[1], 
         label=r''+AuHaloName)
#x, y = sa.dispersionYasX(simStarsData['age'], simStarsData['feh'], nbins=10)
#ax1.plot(x, y, ':', lw=2, rasterized=True, color=color[1])
#ax1.plot(x, y, 'o', ms=3, rasterized=True, c=color[1])

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'Age (Gyr)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$\sigma_{\rm [Fe/H]}$ (dex)', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig.savefig(plotpath + 'sigmaFeHVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate color-magnitude diagram 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
gs = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.5, width_ratios=[2, 1])
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -0.2, 1.8, -2.0, 18.0

# Plot all simulated stars 
ax1.scatter(simStars['jmag'] - simStars['kmag'], simStars['kmag'], 
            rasterized=True, zorder=1, s=0.2, c='lightgrey', label=r''+AuHaloName)

# Draw observed box
bx = Rectangle((jkmin, kmin), jkmax - jkmin, kmax - kmin, zorder=2, ls='--', lw=2, 
             color='k', fill=False)
ax1.add_patch(bx)

# Plot all observed stars 
ax1.scatter(obsStars['jmag'] - obsStars['kmag'], obsStars['kmag'], 
            rasterized=True, zorder=3, s=0.2, c='yellowgreen', label=r'Observed')

# Plot sub-sampled observed stars 
ax1.scatter(obsStarsData['jmag'] - obsStarsData['kmag'], obsStarsData['kmag'], 
            rasterized=True, zorder=4, s=0.2, c=color[0], label=r'Observed sub-sample')

# Plot sub-sampled simulated stars 
ax1.scatter(simStarsData['jmag'] - simStarsData['kmag'], simStarsData['kmag'], 
            rasterized=True, zorder=5, s=0.2, c=color[1], label=r''+AuHaloName+' sub-sample')

ax1.legend(loc='lower left', fontsize=legs12, markerscale=10, frameon=False)
ax1.set_xlabel(r'$J - K$', fontsize=lab12, labelpad=lpad12)
ax1.set_ylabel(r'$K$', fontsize=lab12, labelpad=lpad12)
ax1.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax1.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor =  sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=5, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0.3, 1.2, 0.0, 1.0

# Plot observed stars 
obsHist = rv_histogram(np.histogram(obsStarsData['jmag'] - obsStarsData['kmag'], 
                       bins=100))
x = np.linspace(xmin, xmax, 100)
y = obsHist.cdf(x)
ax2.plot(x, y, '-', lw=2, rasterized=True, zorder=1, c=color[0])

# Plot simulated stars 
simHist = rv_histogram(np.histogram(simStarsData['jmag'] - simStarsData['kmag'], 
                       bins=100))
x = np.linspace(xmin, xmax, 100)
y = simHist.cdf(x)
ax2.plot(x, y, '--', lw=2, rasterized=True, zorder=2, c=color[1])

ax2.set_xlabel(r'$J - K$', fontsize=lab12, labelpad=lpad12)
ax2.set_ylabel(r'CDF', fontsize=lab12, labelpad=lpad12)
ax2.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax2.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=4, 
                                                 nxminor=5, nymajor=3, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 5.0, 13.0, 0.0, 1.0

# Plot observed stars 
obsHist = rv_histogram(np.histogram(obsStarsData['kmag'], bins=100))
x = np.linspace(xmin, xmax, 100)
y = obsHist.cdf(x)
ax3.plot(x, y, '-', lw=2, rasterized=True, zorder=1, c=color[0])

# Plot simulated stars 
simHist = rv_histogram(np.histogram(simStarsData['kmag'], bins=100))
x = np.linspace(xmin, xmax, 100)
y = simHist.cdf(x)
ax3.plot(x, y, '--', lw=2, rasterized=True, zorder=2, c=color[1])

ax3.set_xlabel(r'$K$', fontsize=lab12, labelpad=lpad12)
ax3.set_ylabel(r'CDF', fontsize=lab12, labelpad=lpad12)
ax3.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax3.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=4, 
                                                 nxminor=5, nymajor=3, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax4 = fig.add_subplot(gs[2, 1])
ax4.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0.0, 2.0, 0.0, 1.0

# Plot observed stars 
obsHist = rv_histogram(np.histogram(obsStarsData['plx'], bins=100))
x = np.linspace(xmin, xmax, 100)
y = obsHist.cdf(x)
ax4.plot(x, y, '-', lw=2, rasterized=True, zorder=1, c=color[0])

# Plot simulated stars 
simHist = rv_histogram(np.histogram(simStarsData['plx'], bins=100))
x = np.linspace(xmin, xmax, 100)
y = simHist.cdf(x)
ax4.plot(x, y, '--', lw=2, rasterized=True, zorder=2, c=color[1])

ax4.set_xlabel(r'$\varpi$ (mas)', fontsize=lab12, labelpad=lpad12)
ax4.set_ylabel(r'CDF', fontsize=lab12, labelpad=lpad12)
ax4.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax4.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=4, 
                                                 nxminor=5, nymajor=3, nyminor=5)
    ax4.set_xlim(left=xmin, right=xmax)
    ax4.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax4.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax4.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax4.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax4.yaxis.set_major_locator(majLoc)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'cmd_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate parallax and distance distributions
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.35)

ax1 = fig.add_subplot(211)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = .0, 2.0, 0.0, 2.5

# Plot observed stars
ax1.hist(obsStarsData['plx'], bins=np.linspace(xmin, xmax, 40), color=color[0], 
         density=True, zorder=1, label=r'Observed') 

# Plot simulated stars 
ax1.hist(simStarsData['plx'], bins=np.linspace(xmin, xmax, 40), color=color[1], 
         density=True, alpha=0.8, zorder=2, label=r''+AuHaloName)

ax1.legend(loc='upper right', fontsize=legs12, frameon=False)
ax1.set_xlabel(r'$\pi \ ({\rm mas})$', fontsize=lab12, labelpad=lpad12)
ax1.set_ylabel(r'Probability density', fontsize=lab12, labelpad=lpad12)
ax1.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax1.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2 = fig.add_subplot(212)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 6.0, 0.0, 0.8

# Plot observed stars
ax2.hist(obsStarsData['dist'], bins=np.linspace(xmin, xmax, 40), color=color[0], 
         density=True, zorder=1) 

# Plot simulated stars
ax2.hist(simStarsData['dist'], bins=np.linspace(xmin, xmax, 40), color=color[1], 
         density=True, alpha=0.8, zorder=2)

ax2.set_xlabel(r'$d \ ({\rm kpc})$', fontsize=lab12, labelpad=lpad12)
ax2.set_ylabel(r'Probability density', fontsize=lab12, labelpad=lpad12)
ax2.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax2.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'plxAndDist_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate vertical height distributions
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)

ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = .0, 1.7, 0.0, 2.4

# Plot observed stars
ax1.hist(obsPos['z'], bins=np.linspace(xmin, xmax, 40), color=color[0], density=True, 
         zorder=1, label=r'Observed') 

# Plot simulated stars 
ax1.hist(simPos['z'], bins=np.linspace(xmin, xmax, 40), color=color[1], density=True, 
         alpha=0.8, zorder=2, label=r''+AuHaloName)

ax1.legend(loc='upper right', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'$z$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'Probability density', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'height_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate radial migration distribution
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)

ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -10., 10., 0., 0.50

ax1.hist(simPos['rho'][simStarsData['flag']==-1]-birthPos['rho'][simStarsData['flag']==-1], 
         bins=np.linspace(xmin, xmax, 40), color=color[1], density=True, 
         label=r''+AuHaloName)

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'Probability density', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig.savefig(plotpath + 'migration_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate radial migration vs. age plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14., -10., 10. 

ax1.scatter(simStarsData['age'][simStarsData['flag']==-1], 
    simPos['rho'][simStarsData['flag']==-1] - birthPos['rho'][simStarsData['flag']==-1], 
    rasterized=True, zorder=2, s=0.2, c=color[1], label=r''+AuHaloName)
#data = sa.binMeanStd(simStarsData['age'][simStarsData['flag']==-1], 
#                   simrho[simStarsData['flag']==-1] - rhoBirth[simStarsData['flag']==-1], 
#                   nbins=10)
#ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
#ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'Age (Gyr)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'migrationVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate radial migration vs. position plot 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.35)
ax1 = fig.add_subplot(311)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 20., -10., 10.

ax1.scatter(birthPos['rho'][simStarsData['flag']==-1], 
    simPos['rho'][simStarsData['flag']==-1] - birthPos['rho'][simStarsData['flag']==-1], 
    rasterized=True, s=0.2, c=color[1])

ax1.set_xlabel(r'$R_{\rm birth}$ (kpc)', fontsize=lab13, labelpad=lpad13)
ax1.set_ylabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab13, labelpad=lpad13)
ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2 = fig.add_subplot(312)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -3.3, 3.3, -10., 10.

ax2.scatter(birthPos['phi'][simStarsData['flag']==-1], 
    simPos['rho'][simStarsData['flag']==-1] - birthPos['rho'][simStarsData['flag']==-1], 
    rasterized=True, s=0.2, c=color[1], label=r''+AuHaloName)

ax2.legend(loc='lower left', fontsize=legs13, frameon=False)
ax2.set_xlabel(r'$\phi_{\rm birth}$ (Radian)', fontsize=lab13, labelpad=lpad13)
ax2.set_ylabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab13, labelpad=lpad13)
ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax3 = fig.add_subplot(313)
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -5., 5., -10., 10.

ax3.scatter(birthPos['z'][simStarsData['flag']==-1], 
    simPos['rho'][simStarsData['flag']==-1] - birthPos['rho'][simStarsData['flag']==-1], 
    rasterized=True, s=0.2, c=color[1])

ax3.set_xlabel(r'$z_{\rm birth}$ (kpc)', fontsize=lab13, labelpad=lpad13)
ax3.set_ylabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab13, labelpad=lpad13)
ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'migVsPos_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate radial migration vs. velocity components plot 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.35)
ax1 = fig.add_subplot(311)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -300, 300, -10, 10 

ax1.scatter(birthVel['rho'][simStarsData['flag']==-1], 
    simPos['rho'][simStarsData['flag']==-1] - birthPos['rho'][simStarsData['flag']==-1], 
    rasterized=True, s=0.2, c=color[1])

ax1.set_xlabel(r'$V_r \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax1.set_ylabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab13, labelpad=lpad13)
ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2 = fig.add_subplot(312)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -400, 100, -10, 10

ax2.scatter(birthVel['phi'][simStarsData['flag']==-1], 
    simPos['rho'][simStarsData['flag']==-1] - birthPos['rho'][simStarsData['flag']==-1], 
    rasterized=True, s=0.2, c=color[1], label=r''+AuHaloName)

ax2.legend(loc='lower left', fontsize=legs13, frameon=False)
ax2.set_xlabel(r'$V_{\phi} \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax2.set_ylabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab13, labelpad=lpad13)
ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax3 = fig.add_subplot(313)
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -200, 200, -10, 10

ax3.scatter(birthVel['z'][simStarsData['flag']==-1], 
    simPos['rho'][simStarsData['flag']==-1] - birthPos['rho'][simStarsData['flag']==-1], 
    rasterized=True, s=0.2, c=color[1])

ax3.set_xlabel(r'$V_z \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax3.set_ylabel(r'$R - R_{\rm birth} \ ({\rm kpc})$', fontsize=lab13, labelpad=lpad13)
ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'migVsVel_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate logg vs. Teff plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 3900, 5400, 1.0, 3.6 

# Plot observed stars 
ax1.scatter(obsStarsData['teff'], obsStarsData['logg'], rasterized=True, zorder=1, 
            s=0.2, c=color[0], label=r'Observed')

# Plot simulated stars 
ax1.scatter(simStarsData['teff'], simStarsData['logg'], rasterized=True, zorder=2, 
            s=0.2, c=color[1], label=r''+AuHaloName)

ax1.legend(loc='upper left', fontsize=legs1, markerscale=10, frameon=False)
ax1.set_xlabel(r'$T_{\rm eff}$ (K)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$\log \ g$', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmax, right=xmin)
    ax1.set_ylim(bottom=ymax, top=ymin)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'kiel_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate [a/Fe] vs. [Fe/H] plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -1.5, 1.0, -0.1, 0.4

# Plot observed stars 
ax1.scatter(obsStarsData['feh'], obsStarsData['afe'], rasterized=True, zorder=1, 
            s=0.2, c=color[0], label=r'Observed')

# Plot simulated stars 
ax1.scatter(simStarsData['feh'], simStarsData['afe'], rasterized=True, zorder=2, 
            s=0.2, c=color[1], label=r''+AuHaloName)

# Plot zero levels
ax1.plot([xmin, xmax], [0., 0.], ls='--', lw=1, c='black', zorder=3)
ax1.plot([0, 0], [ymin, ymax], ls='--', lw=1, c='black', zorder=4)

ax1.legend(loc='upper right', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'[Fe/H]', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'[$\alpha$/Fe]', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig.savefig(plotpath + 'afeVsFeh_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# [alpha/Fe] and [Fe/H] vs. Age 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.10)

ax1 = fig.add_subplot(211)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14.0, -0.1, 0.4

# Plot observed stars 
ax1.scatter(obsStarsData['age'], obsStarsData['afe'], rasterized=True, zorder=1, 
            s=0.2, c=color[0])
#data = sa.binMeanStd(obsStarsData['age'], obsStarsData['afe'], nbins=20)
#ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
#ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=4)

# Plot simulated stars 
#ageErr = 0.15 * simStarsData['age']
#simAge = np.random.normal(loc=simStarsData['age'], scale=ageErr, 
#                          size=(1, len(simStarsData['age'])))
#ax1.scatter(simAge[0, :], simStarsData['afe'], rasterized=True, zorder=2, 
#            s=0.2, c=color[1])
ax1.scatter(simStarsData['age'], simStarsData['afe'], rasterized=True, zorder=2, 
            s=0.2, c=color[1])

# Plot zero level
ax1.plot([0,25], [0,0], ls='--', lw=1, c='black', zorder=5)

ax1.set_ylabel(r'[$\alpha$/Fe]', fontsize=lab12, labelpad=lpad12)
ax1.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax1.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(212)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14.0, -1.5, 1.0

# Plot observed stars 
ax2.scatter(obsStarsData['age'], obsStarsData['feh'], rasterized=True, zorder=1, 
            s=0.2, c=color[0], label=r'Observed')

# Plot simulated stars 
#ax2.scatter(simAge[0, :], simStarsData['feh'], rasterized=True, zorder=2, s=0.2, c=color[1], 
#            label=r''+AuHaloName)
ax2.scatter(simStarsData['age'], simStarsData['feh'], rasterized=True, zorder=2, 
            s=0.2, c=color[1], label=r''+AuHaloName)

# Plot zero level
ax2.plot([0,25], [0,0], ls='--', lw=1, c='black', zorder=3)

ax2.legend(loc='lower left', fontsize=legs12, frameon=False)
ax2.set_xlabel(r'Age (Gyr)', fontsize=lab12, labelpad=lpad12)
ax2.set_ylabel(r'[Fe/H]', fontsize=lab12, labelpad=lpad12)
ax2.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax2.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'abundVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate velocity distributions (ICRS)
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.35)
ax1 = fig.add_subplot(311)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -30, 20, 0.00, 0.15

# Plot observed stars 
ax1.hist(obsStarsData['pm_ra_cosdec'], bins=np.linspace(xmin, xmax, 40), color=color[0], 
         density=True, zorder=1, label=r'Observed') 

# Plot simulated stars 
ax1.hist(simStarsData['pm_ra_cosdec'], bins=np.linspace(xmin, xmax, 40), color=color[1], 
         density=True, alpha=0.8, zorder=2, label=r''+AuHaloName)

ax1.legend(loc='upper left', fontsize=legs13, frameon=False)
ax1.set_xlabel(r'$\mu_{\alpha}^* \ ({\rm mas \ yr}^{-1})$', fontsize=lab13, 
               labelpad=lpad13)
ax1.set_ylabel(r'Probability density', fontsize=lab13, labelpad=lpad13)
ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=4, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(312)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -30, 20, 0.00, 0.13

# Plot observed stars
ax2.hist(obsStarsData['pm_dec'], bins=np.linspace(xmin, xmax, 40), color=color[0], 
         density=True, zorder=1) 

# Plot simulated stars 
ax2.hist(simStarsData['pm_dec'], bins=np.linspace(xmin, xmax, 40), color=color[1], 
         density=True, alpha=0.8, zorder=2)

ax2.set_xlabel(r'$\mu_{\delta} \ ({\rm mas \ yr}^{-1})$', fontsize=lab13, 
               labelpad=lpad13)
ax2.set_ylabel(r'Probability density', fontsize=lab13, labelpad=lpad13)
ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=4, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax3 = fig.add_subplot(313)
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -300, 200, 0.00, 0.017

# Plot observed stars
ax3.hist(obsStarsData['rv'], bins=np.linspace(xmin, xmax, 40), color=color[0], 
         density=True, zorder=1) 

# Plot simulated stars
ax3.hist(simStarsData['rv'], bins=np.linspace(xmin, xmax, 40), color=color[1], 
         density=True, alpha=0.8, zorder=2)

# Two sample KS test
#stat, pval = ks_2samp(obsStarsData['rv'], simStarsData['rv'])
#print ('KS statistic, p-value = %.2e, %.2e' %(stat, pval))
#stat, critical_values, significance_level = anderson_ksamp([obsStarsData['rv'], 
#    simStarsData['rv']])
#print ('AD statistic, critical value at 5 percent level, significance level = ' 
#    '%.2e, %.2e, %.2e' %(stat, critical_values[2], significance_level))

ax3.set_xlabel(r'$v_r \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax3.set_ylabel(r'Probability density', fontsize=lab13, labelpad=lpad13)
ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=4, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

fig.savefig(plotpath + 'velocity_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)



# Generate velocity distributions (GC)
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.35)
ax1 = fig.add_subplot(311)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -500, 500, 1e-5, 1e-1

# Plot observed stars 
sns.distplot(obsVel['z'], bins=np.linspace(xmin, xmax, 20), hist=False, kde=True, 
             norm_hist=True, color=color[0], kde_kws={'linestyle': '-'}, ax=ax1) 

# Plot simulated stars 
sns.distplot(simVel['z'], bins=np.linspace(xmin, xmax, 20), hist=False, kde=True, 
             norm_hist=True, color=color[1], kde_kws={'linestyle': '--'}, ax=ax1) 
sns.distplot(simVel['z'][simStarsData['flag']==-1], bins=np.linspace(xmin, xmax, 20), 
             hist=False, kde=True, norm_hist=True, color=color[2], 
             kde_kws={'linestyle': ':'}, ax=ax1)
sns.distplot(simVel['z'][simStarsData['flag']==0], bins=np.linspace(xmin, xmax, 20), 
             hist=False, kde=True, norm_hist=True, color=color[3], 
             kde_kws={'linestyle': '-.'}, ax=ax1)

ax1.set_xlabel(r'$V_z \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax1.set_ylabel(r'Kernel density', fontsize=lab13, labelpad=lpad13)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=4, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax1.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax2 = fig.add_subplot(312)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -500, 500, 1e-5, 1e-0

# Plot observed stars
sns.distplot(obsVel['phi'], bins=np.linspace(xmin, xmax, 20), hist=False, kde=True, 
             norm_hist=True, color=color[0], kde_kws={'linestyle': '-'}, ax=ax2,
             label=r'Observed')

# Plot simulated stars 
sns.distplot(simVel['phi'], bins=np.linspace(xmin, xmax, 20), hist=False, kde=True, 
             norm_hist=True, color=color[1], kde_kws={'linestyle': '--'}, ax=ax2,
             label=r''+AuHaloName)
sns.distplot(simVel['phi'][simStarsData['flag']==-1], bins=np.linspace(xmin, xmax, 20), 
             hist=False, kde=True, norm_hist=True, color=color[2], 
             kde_kws={'linestyle': ':'}, ax=ax2, label=r''+AuHaloName+': in-situ') 
sns.distplot(simVel['phi'][simStarsData['flag']==0], bins=np.linspace(xmin, xmax, 20), 
             hist=False, kde=True, norm_hist=True, color=color[3], 
             kde_kws={'linestyle': '-.'}, ax=ax2, label=r''+AuHaloName+': accreted') 

ax2.legend(loc='upper right', fontsize=legs13, ncol=2, frameon=False)
ax2.set_xlabel(r'$V_{\phi} \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax2.set_ylabel(r'Kernel density', fontsize=lab13, labelpad=lpad13)
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=4, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax2.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax3 = fig.add_subplot(313)
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -500, 500, 1e-5, 1e-1

# Plot observed stars
sns.distplot(obsVel['rho'], bins=np.linspace(xmin, xmax, 40), hist=False, kde=True, 
             norm_hist=True, color=color[0], kde_kws={'linestyle': '-'}, ax=ax3)

# Plot simulated stars 
sns.distplot(simVel['rho'], bins=np.linspace(xmin, xmax, 40), hist=False, kde=True, 
             norm_hist=True, color=color[1], kde_kws={'linestyle': '--'}, ax=ax3)
sns.distplot(simVel['rho'][simStarsData['flag']==-1], bins=np.linspace(xmin, xmax, 40), 
             hist=False, kde=True, norm_hist=True, color=color[2], 
             kde_kws={'linestyle': ':'}, ax=ax3) 
sns.distplot(simVel['rho'][simStarsData['flag']==0], bins=np.linspace(xmin, xmax, 40), 
             hist=False, kde=True, norm_hist=True, color=color[3], 
             kde_kws={'linestyle': '-.'}, ax=ax3) 

ax3.set_xlabel(r'$V_r \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax3.set_ylabel(r'Kernel density', fontsize=lab13, labelpad=lpad13)
ax3.set_yscale('log')
ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
                                                 nxminor=5, nymajor=4, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax3.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

fig.savefig(plotpath + 'velocityGC_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate velocity vs. Age plot 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.10)
ax1 = fig.add_subplot(311)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 14, -400, 400 

# Plot observed stars 
ax1.scatter(obsStarsData['age'], obsVel['z'], rasterized=True, zorder=1, s=0.2, 
            c=color[0])

# Plot simulated stars 
#ax1.scatter(simStarsData['age'][simzmax<2.], simvz[simzmax<2.], rasterized=True, 
#            zorder=2, s=0.2, c=color[1])
#ax1.scatter(simStarsData['age'][simzmax>=2.], simvz[simzmax>=2.], rasterized=True, zorder=3,
#            s=0.2, c=color[2])
ax1.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            simVel['z'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1])
ax1.scatter(simStarsData['age'][simStarsData['flag']==0], 
            simVel['z'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2])

ax1.set_ylabel(r'$V_z \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax2 = fig.add_subplot(312)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 14, -400, 400 

# Plot observed stars 
ax2.scatter(obsStarsData['age'], obsVel['phi'], rasterized=True, zorder=1, s=0.2, 
            c=color[0], label=r'Observed')

# Plot simulated stars 
#ax2.scatter(simStarsData['age'][simzmax<2.], simvp[simzmax<2.], rasterized=True, 
#            zorder=2, s=0.2, c=color[1], label=r''+AuHaloName+r' ($z_{\rm max} < 2$kpc)')
#ax2.scatter(simStarsData['age'][simzmax>=2.], simvp[simzmax>=2.], rasterized=True, zorder=3,
#            s=0.2, c=color[2], label=r''+AuHaloName+r' ($z_{\rm max} \geq 2$kpc)')
ax2.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            simVel['phi'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1], label=r''+AuHaloName+': in-situ')
ax2.scatter(simStarsData['age'][simStarsData['flag']==0], 
            simVel['phi'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2], label=r''+AuHaloName+': accreted')

ax2.legend(loc='upper left', fontsize=legs13, frameon=False)
ax2.set_ylabel(r'$V_{\phi} \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax3 = fig.add_subplot(313)
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 14, -400, 400 

# Plot observed stars 
ax3.scatter(obsStarsData['age'], obsVel['rho'], rasterized=True, zorder=1, s=0.2, 
            c=color[0])

# Plot simulated stars 
#ax3.scatter(simStarsData['age'][simzmax<2.], simvr[simzmax<2.], rasterized=True, 
#            zorder=2, s=0.2, c=color[1])
#ax3.scatter(simStarsData['age'][simzmax>=2.], simvr[simzmax>=2.], rasterized=True, zorder=3,
#            s=0.2, c=color[2])
ax3.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            simVel['rho'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1])
ax3.scatter(simStarsData['age'][simStarsData['flag']==0], 
            simVel['rho'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2])

ax3.set_xlabel(r'Age (Gyr)', fontsize=lab13, labelpad=lpad13)
ax3.set_ylabel(r'$V_r \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))

fig.savefig(plotpath + 'velocityVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate velocity vs. Age plot (color coded with birth radius) 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.10)
ax1 = fig.add_subplot(311)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 14, -400, 400 

# Plot observed stars 
#ax1.scatter(obsStarsData['age'], obsvz, rasterized=True, zorder=1, s=0.2, c=color[0])

# Plot simulated stars 
im = ax1.scatter(simStarsData['age'][birthPos['rho']<15.],simVel['z'][birthPos['rho']<15.], 
                 c=simPos['rho'][birthPos['rho']<15.] - birthPos['rho'][birthPos['rho']<15.], 
                 cmap='PiYG', rasterized=True, zorder=2, s=0.2)
clb = plt.colorbar(im, ticks=MaxNLocator(5), format='%d')
clb.set_label(r'$R - R_{\rm birth}$ (kpc)')

ax1.set_ylabel(r'$V_z \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax2 = fig.add_subplot(312)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 14, -400, 400 

# Plot observed stars 
#ax2.scatter(obsStarsData['age'], obsvp, rasterized=True, zorder=1, s=0.2, c=color[0], 
#            label=r'Observed')

# Plot simulated stars 
im = ax2.scatter(simStarsData['age'][birthPos['rho']<15.],simVel['phi'][birthPos['rho']<15.], 
                 c=simPos['rho'][birthPos['rho']<15.] - birthPos['rho'][birthPos['rho']<15.], 
                 cmap='PiYG', rasterized=True, zorder=2, s=0.2, 
                 label=r''+AuHaloName+r' ($R_{\rm birth} < 15.$ kpc)')
clb = plt.colorbar(im, ticks=MaxNLocator(5), format='%d')
clb.set_label(r'$R - R_{\rm birth}$ (kpc)')


ax2.legend(loc='upper left', fontsize=legs13, frameon=False)
ax2.set_ylabel(r'$V_{\phi} \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax3 = fig.add_subplot(313)
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 14, -400, 400 

# Plot observed stars 
#ax3.scatter(obsStarsData['age'], obsvr, rasterized=True, zorder=1, s=0.2, c=color[0])

# Plot simulated stars 
im = ax3.scatter(simStarsData['age'][birthPos['rho']<15.], simVel['rho'][birthPos['rho']<15.], 
                 c=simPos['rho'][birthPos['rho']<15.] - birthPos['rho'][birthPos['rho']<15.], 
                 cmap='PiYG', rasterized=True, zorder=2, s=0.2)
clb = plt.colorbar(im, ticks=MaxNLocator(5), format='%d')
clb.set_label(r'$R - R_{\rm birth}$ (kpc)')


ax3.set_xlabel(r'Age (Gyr)', fontsize=lab13, labelpad=lpad13)
ax3.set_ylabel(r'$V_r \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))

fig.savefig(plotpath + 'velocityVsAgeVsRB_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate velocity vs. distance plot 
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.10)
ax1 = fig.add_subplot(311)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 6, -400, 400 

# Plot observed stars 
ax1.scatter(obsStarsData['dist'], obsVel['z'], rasterized=True, zorder=1, s=0.2, 
            c=color[0])

# Plot simulated stars 
#ax1.scatter(simStarsData['dist'][simzmax<2.], simvz[simzmax<2.], rasterized=True, 
#            zorder=2, s=0.2, c=color[1])
#ax1.scatter(simStarsData['dist'][simzmax>=2.], simvz[simzmax>=2.], rasterized=True, zorder=3,
#            s=0.2, c=color[2])
ax1.scatter(simStarsData['dist'][simStarsData['flag']==-1], 
            simVel['z'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1])
ax1.scatter(simStarsData['dist'][simStarsData['flag']==0], 
            simVel['z'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, c=color[2])

ax1.set_ylabel(r'$V_z \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax2 = fig.add_subplot(312)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 6, -400, 400 

# Plot observed stars 
ax2.scatter(obsStarsData['dist'], obsVel['phi'], rasterized=True, zorder=1, s=0.2, 
            c=color[0], label=r'Observed')

# Plot simulated stars 
#ax2.scatter(simStarsData['dist'][simzmax<2.], simvp[simzmax<2.], rasterized=True, 
#            zorder=2, s=0.2, c=color[1], label=r''+AuHaloName+r' ($z_{\rm max} < 2$kpc)')
#ax2.scatter(simStarsData['dist'][simzmax>=2.], simvp[simzmax>=2.], rasterized=True, zorder=3,
#            s=0.2, c=color[2], label=r''+AuHaloName+r' ($z_{\rm max} \geq 2$kpc)')
ax2.scatter(simStarsData['dist'][simStarsData['flag']==-1], 
            simVel['phi'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1], label=r''+AuHaloName+': in-situ')
ax2.scatter(simStarsData['dist'][simStarsData['flag']==0], 
            simVel['phi'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2], label=r''+AuHaloName+': accreted')

ax2.legend(loc='upper left', fontsize=legs13, frameon=False)
ax2.set_ylabel(r'$V_{\phi} \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

ax3 = fig.add_subplot(313)
ax3.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0, 6, -400, 400 

# Plot observed stars 
ax3.scatter(obsStarsData['dist'], obsVel['rho'], rasterized=True, zorder=1, s=0.2, 
            c=color[0])

# Plot simulated stars 
#ax3.scatter(simStarsData['dist'][simzmax<2.], simvr[simzmax<2.], rasterized=True, 
#            zorder=2, s=0.2, c=color[1])
#ax3.scatter(simStarsData['dist'][simzmax>=2.], simvr[simzmax>=2.], rasterized=True, zorder=3,
#            s=0.2, c=color[2])
ax3.scatter(simStarsData['dist'][simStarsData['flag']==-1], 
            simVel['rho'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1])
ax3.scatter(simStarsData['dist'][simStarsData['flag']==0], 
            simVel['rho'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2])

ax3.set_xlabel(r'$d$ (kpc)', fontsize=lab13, labelpad=lpad13)
ax3.set_ylabel(r'$V_r \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
                pad=tpad13)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax3.set_xlim(left=xmin, right=xmax)
    ax3.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax3.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax3.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax3.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax3.yaxis.set_major_locator(majLoc)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))

fig.savefig(plotpath + 'velocityVsdist_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate z vs. R plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 7.3, 8.7, 0., 1.4 

# Plot observed stars 
ax1.scatter(obsPos['rho'], obsPos['z'], rasterized=True, zorder=1, s=0.2, c=color[0], 
            label=r'Observed')

# Plot simulated stars 
ax1.scatter(simPos['rho'], simPos['z'], rasterized=True, zorder=2, s=0.2, c=color[1], 
            label=r''+AuHaloName)
#ax1.scatter(simrho[simStarsData['flag']==-1], simz[simStarsData['flag']==-1], 
#            rasterized=True, zorder=2, s=0.2, c=color[1], label=r''+AuHaloName+': in-situ')
#ax1.scatter(simrho[simStarsData['flag']==0], simz[simStarsData['flag']==0], 
#            rasterized=True, zorder=3, s=0.2, c=color[2], label=r''+AuHaloName+': accreted')

ax1.legend(loc='lower right', fontsize=legs1, markerscale=10, frameon=False)
ax1.set_xlabel(r'$R$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$z$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'heightVsradius_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Radial coordinate, height vs. age
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
                    hspace=0.10)

ax1 = fig.add_subplot(211)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14., 7.5, 9.5 

# Plot observed stars
ax1.scatter(obsStarsData['age'], obsPos['rho'], rasterized=True, zorder=1, s=0.2, 
            c=color[0]) 
data = sa.binMeanStd(obsStarsData['age'], obsPos['rho'], nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=5)

# Plot simulated stars 
ax1.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            simPos['rho'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1])
ax1.scatter(simStarsData['age'][simStarsData['flag']==0], 
            simPos['rho'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2])
data = sa.binMeanStd(simStarsData['age'], simPos['rho'], nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax1.set_ylabel(r'$R$ (kpc)', fontsize=lab12, labelpad=lpad12)
ax1.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax1.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12, labelbottom=False)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2 = fig.add_subplot(212)
ax2.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14., 0., 2. 

# Plot observed stars
ax2.scatter(obsStarsData['age'], obsPos['z'], rasterized=True, zorder=1, s=0.2, 
            c=color[0], label=r'Observed') 
data = sa.binMeanStd(obsStarsData['age'], obsPos['z'], nbins=20)
ax2.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
ax2.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=5)

# Plot simulated stars 
ax2.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            simPos['z'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1], label=r''+AuHaloName+': in-situ')
ax2.scatter(simStarsData['age'][simStarsData['flag']==0], 
            simPos['z'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2], label=r''+AuHaloName+': accreted')
data = sa.binMeanStd(simStarsData['age'], simPos['z'], nbins=20)
ax2.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
ax2.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax2.legend(loc='upper left', fontsize=legs12, frameon=False)
ax2.set_xlabel(r'Age (Gyr)', fontsize=lab12, labelpad=lpad12)
ax2.set_ylabel(r'$z \ ({\rm kpc})$', fontsize=lab12, labelpad=lpad12)
ax2.tick_params(axis='y', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
ax2.tick_params(axis='x', labelsize=tick12, which='both', direction='inout',
                pad=tpad12)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=5, nyminor=5)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax2.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax2.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax2.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax2.yaxis.set_major_locator(majLoc)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'coordinateVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate guiding radius vs. age plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14., 0., 18. #None, None, None, None

# Plot observed stars 
ax1.scatter(obsStarsData['age'], np.abs(obsAct['phi']/220.), rasterized=True, zorder=1, 
            s=0.2, c=color[0], label=r'Observed')
data = sa.binMeanStd(obsStarsData['age'], np.abs(obsAct['phi']/220.), nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=5)

# Plot simulated stars 
ax1.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            np.abs(simAct['phi'][simStarsData['flag']==-1]/220.), rasterized=True, 
            zorder=2, s=0.2, c=color[1], label=r''+AuHaloName+': in-situ')
ax1.scatter(simStarsData['age'][simStarsData['flag']==0], 
            np.abs(simAct['phi'][simStarsData['flag']==0]/220.), rasterized=True, 
            zorder=3, s=0.2, c=color[2], label=r''+AuHaloName+': accreted')
data = sa.binMeanStd(simStarsData['age'], np.abs(simAct['phi']/220.), nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'Age (Gyr)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$R_{\rm guide}$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'rguideVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate guiding radius vs. distance plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 6., 0., 18. #None, None, None, None

# Plot observed stars 
ax1.scatter(obsStarsData['dist'], np.abs(obsAct['phi']/220.), rasterized=True, zorder=1, 
            s=0.2, c=color[0], label=r'Observed')
#data = sa.binMeanStd(obsStarsData['dist'], np.abs(obslz/220.), nbins=20)
#ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
#ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=5)

# Plot simulated stars 
ax1.scatter(simStarsData['dist'][simStarsData['flag']==-1], 
            np.abs(simAct['phi'][simStarsData['flag']==-1]/220.), rasterized=True, 
            zorder=2, s=0.2, c=color[1], label=r''+AuHaloName+': in-situ')
ax1.scatter(simStarsData['dist'][simStarsData['flag']==0], 
            np.abs(simAct['phi'][simStarsData['flag']==0]/220.), rasterized=True, 
            zorder=3, s=0.2, c=color[2], label=r''+AuHaloName+': accreted')
#data = sa.binMeanStd(simStarsData['dist'], np.abs(simlz/220.), nbins=20)
#ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
#ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'$d$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$R_{\rm guide}$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'rguideVsdist_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate eccentricity vs. age plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14., 0.0, 1. #None, None, None, None

# Plot observed stars 
ax1.scatter(obsStarsData['age'], obsOrb['ecc'], rasterized=True, zorder=1, s=0.2, 
            c=color[0], label=r'Observed')
data = sa.binMeanStd(obsStarsData['age'], obsOrb['ecc'], nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=5)

# Plot simulated stars 
#ax1.scatter(simStarsData['age'], simecc, rasterized=True, zorder=2, s=0.2, c=color[1], 
#            label=r''+AuHaloName)
ax1.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            simOrb['ecc'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1], label=r''+AuHaloName+': in-situ')
ax1.scatter(simStarsData['age'][simStarsData['flag']==0], 
            simOrb['ecc'][simStarsData['flag']==0], rasterized=True, zorder=2, s=0.2, 
            c=color[2], label=r''+AuHaloName+': accreted')
data = sa.binMeanStd(simStarsData['age'], simOrb['ecc'], nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'Age (Gyr)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$e$', fontsize=lab1, labelpad=lpad1)
#ax1.set_yscale('log')
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=6, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.savefig(plotpath + 'eccVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate maximum height vs. age plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 14., 0.01, 100 #None, None, None, None

# Plot observed stars 
ax1.scatter(obsStarsData['age'], obsOrb['zmax'], rasterized=True, zorder=1, s=0.2, 
            c=color[0], label=r'Observed')
data = sa.binMeanStd(obsStarsData['age'], obsOrb['zmax'], nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=5)

# Plot simulated stars 
#ax1.scatter(simStarsData['age'], simzmax, rasterized=True, zorder=2, s=0.2, c=color[1], 
#            label=r''+AuHaloName)
ax1.scatter(simStarsData['age'][simStarsData['flag']==-1], 
            simOrb['zmax'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1], label=r''+AuHaloName+': in-situ')
ax1.scatter(simStarsData['age'][simStarsData['flag']==0], 
            simOrb['zmax'][simStarsData['flag']==0], rasterized=True, zorder=2, s=0.2, 
            c=color[2], label=r''+AuHaloName+': accreted')
data = sa.binMeanStd(simStarsData['age'], simOrb['zmax'], nbins=20)
ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'Age (Gyr)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$Z_{\rm max}$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
fig.savefig(plotpath + 'zmaxVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate maximum height vs. distance plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = 0., 6., 0.01, 100 #None, None, None, None

# Plot observed stars 
ax1.scatter(obsStarsData['dist'], obsOrb['zmax'], rasterized=True, zorder=1, s=0.2, 
            c=color[0], label=r'Observed')
#data = sa.binMeanStd(obsStarsData['dist'], obszmax, nbins=20)
#ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=3)
#ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[0], mec='k', zorder=5)

# Plot simulated stars 
ax1.scatter(simStarsData['dist'][simStarsData['flag']==-1], 
            simOrb['zmax'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1], label=r''+AuHaloName+': in-situ')
ax1.scatter(simStarsData['dist'][simStarsData['flag']==0], 
            simOrb['zmax'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2], label=r''+AuHaloName+': accreted')
#data = sa.binMeanStd(simStarsData['dist'], simzmax, nbins=20)
#ax1.plot(data[:, 0], data[:, 1], '-', lw=2, color='k', zorder=4)
#ax1.plot(data[:, 0], data[:, 1], 'o', ms=3, color=color[1], mec='k', zorder=6)

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'$d$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$Z_{\rm max}$ (kpc)', fontsize=lab1, labelpad=lpad1)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
fig.savefig(plotpath + 'zmaxVsdist_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate vertical height vs. angular momentum plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -3000., 3000., 0., 2.0 #None, None, None, None 

# Plot observed stars
ax1.scatter(obsAct['phi'], obsPos['z'], rasterized=True, zorder=1, s=0.2, c=color[0], 
            label=r'Observed') 

# Plot simulated stars 
ax1.scatter(simAct['phi'][simStarsData['flag']==-1], simPos['z'][simStarsData['flag']==-1], 
            rasterized=True, zorder=2, s=0.2, c=color[1], label=r''+AuHaloName+': in-situ')
ax1.scatter(simAct['phi'][simStarsData['flag']==0], simPos['z'][simStarsData['flag']==0], 
            rasterized=True, zorder=3, s=0.2, c=color[2], label=r''+AuHaloName+': accreted')

ax1.legend(loc='upper right', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'$L_z \ ({\rm kpc \ km \ s}^{-1})$', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$z \ ({\rm kpc})$', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig(plotpath + 'heightVsLz_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Generate radial action vs. angular momentum plot
#-----------------------------------------------------------------------------------------
fig = plt.figure()
sns.set(rc={'text.usetex' : True})
sns.set_style("ticks")
fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
                    hspace=0.08)
ax1 = fig.add_subplot(111)
ax1.set_rasterization_zorder(-1)
xmin, xmax, ymin, ymax = -3000., 3000, 0., 3000. #None, None, None, None 

# Plot observed stars
ax1.scatter(obsAct['phi'], obsAct['rho'], rasterized=True, zorder=1, s=0.2, c=color[0], 
            label=r'Observed') 

# Plot simulated stars 
ax1.scatter(simAct['phi'][simStarsData['flag']==-1], 
            simAct['rho'][simStarsData['flag']==-1], rasterized=True, zorder=2, s=0.2, 
            c=color[1], label=r''+AuHaloName+': in-situ')
ax1.scatter(simAct['phi'][simStarsData['flag']==0], 
            simAct['rho'][simStarsData['flag']==0], rasterized=True, zorder=3, s=0.2, 
            c=color[2], label=r''+AuHaloName+': accreted')

ax1.legend(loc='upper left', fontsize=legs1, frameon=False)
ax1.set_xlabel(r'$L_z \ ({\rm kpc \ km \ s}^{-1})$', fontsize=lab1, labelpad=lpad1)
ax1.set_ylabel(r'$J_r \ ({\rm kpc \ km \ s}^{-1})$', fontsize=lab1, labelpad=lpad1)
ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
                pad=tpad1)
if xmin is not None:
    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
                                                 nxminor=5, nymajor=7, nyminor=5)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_ylim(bottom=ymin, top=ymax)
    minLoc = MultipleLocator(xminor)
    ax1.xaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(xmajor)
    ax1.xaxis.set_major_locator(majLoc)
    minLoc = MultipleLocator(yminor)
    ax1.yaxis.set_minor_locator(minLoc)
    majLoc = MultipleLocator(ymajor)
    ax1.yaxis.set_major_locator(majLoc)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

fig.savefig(plotpath + 'JrVsLz_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# The following plots were made for certain testing purposes
#-----------------------------------------------------------
#
###########################################################################################
## Generate velocity distributions (GC)
###########################################################################################
#fig = plt.figure()
#sns.set(rc={'text.usetex' : True})
#sns.set_style("ticks")
#fig.subplots_adjust(bottom=0.12, right=0.85, top=0.98, left=0.30, wspace=0.35, 
#                    hspace=0.35)
#ax1 = fig.add_subplot(311)
#ax1.set_rasterization_zorder(-1)
#xmin, xmax, ymin, ymax = -200, 200, 0.00, 0.025
#
## Plot observed stars 
#ax1.hist(obsvz, bins=np.linspace(xmin, xmax, 40), color=color[0], density=True, zorder=1, 
#         label=r'Observed') 
#
## Plot simulated stars 
#ax1.hist(simvz, bins=np.linspace(xmin, xmax, 40), color=color[1], density=True, alpha=0.8, 
#         zorder=2, label=r''+AuHaloName)
#
## Plot simulated stars 
#ax1.hist(simvz[simStarsData['flag']==-1], bins=np.linspace(xmin, xmax, 40), color=color[2], 
#         density=True, alpha=0.8, zorder=3, label=r''+AuHaloName+': in-situ')
#ax1.hist(simvz[simStarsData['flag']==0], bins=np.linspace(xmin, xmax, 40), color=color[3], 
#         density=True, alpha=0.8, zorder=4, label=r''+AuHaloName+': accreted')
#
#ax1.legend(loc='upper left', fontsize=legs13, frameon=False)
#ax1.set_xlabel(r'$V_z \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
#ax1.set_ylabel(r'Probability density', fontsize=lab13, labelpad=lpad13)
#ax1.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
#                pad=tpad13)
#ax1.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
#                pad=tpad13)
#if xmin is not None:
#    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
#                                                 nxminor=5, nymajor=4, nyminor=5)
#    ax1.set_xlim(left=xmin, right=xmax)
#    ax1.set_ylim(bottom=ymin, top=ymax)
#    minLoc = MultipleLocator(xminor)
#    ax1.xaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(xmajor)
#    ax1.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax1.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax1.yaxis.set_major_locator(majLoc)
#    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#
#ax2 = fig.add_subplot(312)
#ax2.set_rasterization_zorder(-1)
#xmin, xmax, ymin, ymax = -400, 0, 0.00, 0.020
#
## Plot observed stars
#ax2.hist(obsvp, bins=np.linspace(xmin, xmax, 40), color=color[0], density=True, zorder=1) 
#
## Plot simulated stars 
#ax2.hist(simvp, bins=np.linspace(xmin, xmax, 40), color=color[1], density=True, alpha=0.8, 
#         zorder=2)
#
## Plot simulated stars 
#ax2.hist(simvp[simStarsData['flag']==-1], bins=np.linspace(xmin, xmax, 40), color=color[2], 
#         density=True, alpha=0.8, zorder=3)
#ax2.hist(simvp[simStarsData['flag']==0], bins=np.linspace(xmin, xmax, 40), color=color[3], 
#         density=True, alpha=0.8, zorder=4)
#
#ax2.set_xlabel(r'$V_{\phi} \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
#ax2.set_ylabel(r'Probability density', fontsize=lab13, labelpad=lpad13)
#ax2.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
#                pad=tpad13)
#ax2.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
#                pad=tpad13)
#if xmin is not None:
#    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
#                                                 nxminor=5, nymajor=4, nyminor=5)
#    ax2.set_xlim(left=xmin, right=xmax)
#    ax2.set_ylim(bottom=ymin, top=ymax)
#    minLoc = MultipleLocator(xminor)
#    ax2.xaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(xmajor)
#    ax2.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax2.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax2.yaxis.set_major_locator(majLoc)
#    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#
#ax3 = fig.add_subplot(313)
#ax3.set_rasterization_zorder(-1)
#xmin, xmax, ymin, ymax = -200, 200, 0.00, 0.015
#
## Plot observed stars
#ax3.hist(obsvr, bins=np.linspace(xmin, xmax, 40), color=color[0], density=True, zorder=1) 
#
## Plot simulated stars
#ax3.hist(simvr, bins=np.linspace(xmin, xmax, 40), color=color[1], density=True, alpha=0.8, 
#         zorder=2)
#
## Plot simulated stars 
#ax3.hist(simvr[simStarsData['flag']==-1], bins=np.linspace(xmin, xmax, 40), color=color[2], 
#         density=True, alpha=0.8, zorder=3)
#ax3.hist(simvr[simStarsData['flag']==0], bins=np.linspace(xmin, xmax, 40), color=color[3], 
#         density=True, alpha=0.8, zorder=4)
#
#ax3.set_xlabel(r'$V_r \ ({\rm km \ s}^{-1})$', fontsize=lab13, labelpad=lpad13)
#ax3.set_ylabel(r'Probability density', fontsize=lab13, labelpad=lpad13)
#ax3.tick_params(axis='y', labelsize=tick13, which='both', direction='inout',
#                pad=tpad13)
#ax3.tick_params(axis='x', labelsize=tick13, which='both', direction='inout',
#                pad=tpad13)
#if xmin is not None:
#    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=6, 
#                                                 nxminor=5, nymajor=4, nyminor=5)
#    ax3.set_xlim(left=xmin, right=xmax)
#    ax3.set_ylim(bottom=ymin, top=ymax)
#    minLoc = MultipleLocator(xminor)
#    ax3.xaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(xmajor)
#    ax3.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax3.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax3.yaxis.set_major_locator(majLoc)
#    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#
#fig.savefig(plotpath + 'velocityGC_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
#plt.close(fig)
#
#
###########################################################################################
## Generate uncertainty distribution for radial velocity
###########################################################################################
#fig = plt.figure()
#sns.set(rc={'text.usetex' : True})
#sns.set_style("ticks")
#fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
#                    hspace=0.08)
#
#ax1 = fig.add_subplot(111)
#ax1.set_rasterization_zorder(-1)
#xmin, xmax, ymin, ymax = .0, 3., 0.0, 1.4
#
#ax1.hist(obsStarsData['rvErr'], bins=np.linspace(xmin, xmax, 40), color=color[0], 
#         density=True) 
#
#ax1.set_xlabel(r'$\sigma_{v_r} \ ({\rm km \ s}^{-1})$', fontsize=lab1, labelpad=lpad1)
#ax1.set_ylabel(r'Probability density', fontsize=lab1, labelpad=lpad1)
#ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
#                pad=tpad1)
#ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
#                pad=tpad1)
#if xmin is not None:
#    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
#                                                 nxminor=5, nymajor=7, nyminor=5)
#    ax1.set_xlim(left=xmin, right=xmax)
#    ax1.set_ylim(bottom=ymin, top=ymax)
#    minLoc = MultipleLocator(xminor)
#    ax1.xaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(xmajor)
#    ax1.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax1.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax1.yaxis.set_major_locator(majLoc)
#    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#
#fig.savefig(plotpath + 'rvErr__' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
#plt.close(fig)
#
#
###########################################################################################
## Generate line-of-sight velocity vs. age plot
###########################################################################################
#fig = plt.figure()
#sns.set(rc={'text.usetex' : True})
#sns.set_style("ticks")
#fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
#                    hspace=0.08)
#ax1 = fig.add_subplot(111)
#ax1.set_rasterization_zorder(-1)
#xmin, xmax, ymin, ymax = 0., 14., -500, 200
#
## Plot observed stars 
#ax1.scatter(obsStarsData['age'], obsStarsData['rv'], rasterized=True, zorder=1, 
#            s=0.2, c=color[0], label=r'Observed')
#
## Plot simulated stars 
#ax1.scatter(simStarsData['age'], simStarsData['rv'], rasterized=True, zorder=2, 
#            s=0.2, c=color[1], label=r''+AuHaloName)
#
#ax1.plot([xmin, xmax], [0., 0.], ls='--', lw=1, c='black', zorder=3)
#
#ax1.legend(loc='lower left', fontsize=legs1, frameon=False)
#ax1.set_xlabel(r'Age (Gyr)', fontsize=lab1, labelpad=lpad1)
#ax1.set_ylabel(r'$V_{\rm los} \ ({\rm km \ s}^{-1})$', fontsize=lab1, labelpad=lpad1)
#ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
#                pad=tpad1)
#ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
#                pad=tpad1)
#if xmin is not None:
#    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
#                                                nxminor=5, nymajor=7, nyminor=5)
#    ax1.set_xlim(left=xmin, right=xmax)
#    ax1.set_ylim(bottom=ymin, top=ymax)
#    minLoc = MultipleLocator(xminor)
#    ax1.xaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(xmajor)
#    ax1.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax1.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax1.yaxis.set_major_locator(majLoc)
#    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
#
#fig.savefig(plotpath + 'losVsAge_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
#plt.close(fig)
#
#
## Generate height vs. age/mass plot 
##-----------------------------------------------------------------------------------------
#fig = plt.figure()
#sns.set(rc={'text.usetex' : True})
#sns.set_style("ticks")
#fig.subplots_adjust(bottom=0.12, right=0.98, top=0.98, left=0.12, wspace=0.08, 
#                    hspace=0.08)
#ax1 = fig.add_subplot(111)
#ax1.set_rasterization_zorder(-1)
#xmin, xmax, ymin, ymax = 0.7, 3., 0., 1.2 
#
## Plot observed stars 
#ax1.scatter(obsStarsData['mass'], obsPos['z'], rasterized=True, zorder=1, 
#            s=0.2, c=color[0], label=r'Observed')
#
## Plot simulated stars 
#ax1.scatter(simStarsData['mass'], simPos['z'], rasterized=True, zorder=2, 
#            s=0.2, c=color[1], label=r''+AuHaloName)
#
#ax1.legend(loc='upper left', fontsize=legs1, markerscale=10, frameon=False)
#ax1.set_xlabel(r'$M \ ({\rm M}_\odot)$', fontsize=lab1, labelpad=lpad1)
#ax1.set_ylabel(r'$z$ (kpc)', fontsize=lab1, labelpad=lpad1)
#ax1.tick_params(axis='y', labelsize=tick1, which='both', direction='inout',
#                pad=tpad1)
#ax1.tick_params(axis='x', labelsize=tick1, which='both', direction='inout',
#                pad=tpad1)
#if xmin is not None:
#    xmajor, xminor, ymajor, yminor = sa.majMinTick(xmin, xmax, ymin, ymax, nxmajor=7, 
#                                                 nxminor=5, nymajor=7, nyminor=5)
#    ax1.set_xlim(left=xmin, right=xmax)
#    ax1.set_ylim(bottom=ymin, top=ymax)
#    minLoc = MultipleLocator(xminor)
#    ax1.xaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(xmajor)
#    ax1.xaxis.set_major_locator(majLoc)
#    minLoc = MultipleLocator(yminor)
#    ax1.yaxis.set_minor_locator(minLoc)
#    majLoc = MultipleLocator(ymajor)
#    ax1.yaxis.set_major_locator(majLoc)
#    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#
#fig.savefig(plotpath + 'heightVsMass_' + AuHaloName + '.png', dpi=400, bbox_inches='tight')
#plt.close(fig)

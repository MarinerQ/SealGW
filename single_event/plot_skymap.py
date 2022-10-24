import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
import ligo.skymap.plot
import astropy_healpix as ah
from ligo.skymap import postprocess
from astropy import units as u
from astropy.coordinates import SkyCoord

if __name__ == "__main__":
    skymap = np.loadtxt('skymap/skymap_paper.txt')[:,1]
    #skymap = tempskymap
    skymap = np.exp(skymap)
    skymap /= sum(skymap)

    npix = len(skymap)
    nside = int(np.sqrt(npix/12.0))
    true_de = -0.408
    true_ra = 3.446


    contour = [50,90]

    #Load-skymap
    #skymap=np.loadtxt('prob.txt')
    nside = ah.npix_to_nside(len(skymap))

    # Convert sky map from probability to probability per square degree.
    deg2perpix = ah.nside_to_pixel_area(nside).to_value(u.deg**2)
    probperdeg2 = skymap / deg2perpix

    #Calculate contour levels
    cls = 100 * postprocess.find_greedy_credible_levels(skymap)

    plt.figure(figsize = (10,6))
    #Initialize skymap grid
    ax = plt.axes(projection='astro hours mollweide')
    ax.grid()
    ax.plot_coord(SkyCoord(true_ra, true_de, unit='rad'), 'x', color='green', markersize=8)

    #Plot skymap with labels
    vmax = probperdeg2.max()
    vmin = probperdeg2.min()
    img = ax.imshow_hpx(probperdeg2, cmap='cylon', nested=True, vmin=vmin, vmax=vmax)
    cs = ax.contour_hpx((cls, 'ICRS'), nested=True, linewidths=0.5, levels=contour,colors='k')
    v = np.linspace(vmin, vmax, 2, endpoint=True)
    cb = plt.colorbar(img, orientation='horizontal', ticks=v, fraction=0.045)
    cb.set_label(r'probability per deg$^2$',fontsize=11)


    text=[]
    pp = np.round(contour).astype(int)
    ii = np.round(np.searchsorted(np.sort(cls), contour) *
                            deg2perpix).astype(int)
    for i, p in zip(ii, pp):
        text.append('{:d}% area: {:,d} deg²'.format(p, i))
    ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')


    savefilename = 'skymap_paper.pdf'
    plt.savefig(savefilename)
    print('Skymap saved to '+ savefilename)


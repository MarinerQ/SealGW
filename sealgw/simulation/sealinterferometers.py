# This file redefines bilby Interferometers, inherited from bilby 2.0.2
# The differences are
# 1. Here ET's response function is aligned with LAL's,
#    so that bilby data simulation is consistent with LAL calculation in C.
#    Apr 4 2023: just found the response functions are aligned... Not sure if Bilby changed or LAL changed...
# 2. Here response function can change with time.

import os
import bilby
from bilby.gw.detector.interferometer import Interferometer
from bilby.gw.detector.networks import InterferometerList, TriangularInterferometer
from bilby.gw.detector.calibration import Recalibrate
from bilby.core.utils import PropertyAccessor, radius_of_earth, logger
from bilby.gw.detector.psd import PowerSpectralDensity

BILBY_ROOT_PATH = bilby.__file__[:-11]  # -11 to eliminate "__init__.py"
from bilby_cython.geometry import (
    get_polarization_tensor,
    three_by_three_matrix_contraction,
    time_delay_from_geocenter,
)
import numpy as np
from ..calculation.localization import lal_et_response_function


class SealInterferometer(Interferometer):
    """Class for the Interferometer for SealGW. ET response function aligned with LAL's."""

    length = PropertyAccessor('geometry', 'length')
    latitude = PropertyAccessor('geometry', 'latitude')
    latitude_radians = PropertyAccessor('geometry', 'latitude_radians')
    longitude = PropertyAccessor('geometry', 'longitude')
    longitude_radians = PropertyAccessor('geometry', 'longitude_radians')
    elevation = PropertyAccessor('geometry', 'elevation')
    x = PropertyAccessor('geometry', 'x')
    y = PropertyAccessor('geometry', 'y')
    xarm_azimuth = PropertyAccessor('geometry', 'xarm_azimuth')
    yarm_azimuth = PropertyAccessor('geometry', 'yarm_azimuth')
    xarm_tilt = PropertyAccessor('geometry', 'xarm_tilt')
    yarm_tilt = PropertyAccessor('geometry', 'yarm_tilt')
    vertex = PropertyAccessor('geometry', 'vertex')
    detector_tensor = PropertyAccessor('geometry', 'detector_tensor')

    duration = PropertyAccessor('strain_data', 'duration')
    sampling_frequency = PropertyAccessor('strain_data', 'sampling_frequency')
    start_time = PropertyAccessor('strain_data', 'start_time')
    frequency_array = PropertyAccessor('strain_data', 'frequency_array')
    time_array = PropertyAccessor('strain_data', 'time_array')
    minimum_frequency = PropertyAccessor('strain_data', 'minimum_frequency')
    maximum_frequency = PropertyAccessor('strain_data', 'maximum_frequency')
    frequency_mask = PropertyAccessor('strain_data', 'frequency_mask')
    frequency_domain_strain = PropertyAccessor('strain_data', 'frequency_domain_strain')
    time_domain_strain = PropertyAccessor('strain_data', 'time_domain_strain')

    def __init__(
        self,
        name,
        power_spectral_density,
        minimum_frequency,
        maximum_frequency,
        length,
        latitude,
        longitude,
        elevation,
        xarm_azimuth,
        yarm_azimuth,
        xarm_tilt=0.0,
        yarm_tilt=0.0,
        calibration_model=Recalibrate(),
    ):
        """
        Instantiate an SealInterferometer object.

        Parameters
        ==========
        name: str
            Interferometer name, e.g., H1.
        power_spectral_density: bilby.gw.detector.PowerSpectralDensity
            Power spectral density determining the sensitivity of the detector.
        minimum_frequency: float
            Minimum frequency to analyse for detector.
        maximum_frequency: float
            Maximum frequency to analyse for detector.
        length: float
            Length of the interferometer in km.
        latitude: float
            Latitude North in degrees (South is negative).
        longitude: float
            Longitude East in degrees (West is negative).
        elevation: float
            Height above surface in metres.
        xarm_azimuth: float
            Orientation of the x arm in degrees North of East.
        yarm_azimuth: float
            Orientation of the y arm in degrees North of East.
        xarm_tilt: float, optional
            Tilt of the x arm in radians above the horizontal defined by
            ellipsoid earth model in LIGO-T980044-08.
        yarm_tilt: float, optional
            Tilt of the y arm in radians above the horizontal.
        calibration_model: Recalibration
            Calibration model, this applies the calibration correction to the
            template, the default model applies no correction.
        """

        super().__init__(
            name,
            power_spectral_density,
            minimum_frequency,
            maximum_frequency,
            length,
            latitude,
            longitude,
            elevation,
            xarm_azimuth,
            yarm_azimuth,
            xarm_tilt,
            yarm_tilt,
            calibration_model,
        )

    def antenna_response(self, ra, dec, time, psi, mode):
        """
        Calculate the antenna response function for a given sky location. ET response function aligned with LAL's.

        See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
        [u, v, w] represent the Earth-frame
        [m, n, omega] represent the wave-frame
        Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

        Parameters
        ==========
        ra: float
            right ascension in radians
        dec: float
            declination in radians
        time: float
            geocentric GPS time
        psi: float
            binary polarisation angle counter-clockwise about the direction of propagation
        mode: str
            polarisation mode (e.g. 'plus', 'cross') or the name of a specific detector.
            If mode == self.name, return 1

        Returns
        =======
        float: The antenna response for the specified mode and time/location

        """
        if mode in ["plus", "cross", "x", "y", "breathing", "longitudinal"]:
            if self.name in ['ET1', 'ET2', 'ET3']:
                return lal_et_response_function(ra, dec, time, psi, self.name, mode)
            else:
                polarization_tensor = get_polarization_tensor(ra, dec, time, psi, mode)
                return three_by_three_matrix_contraction(
                    self.geometry.detector_tensor, polarization_tensor
                )
        elif mode == self.name:
            return 1
        else:
            return 0


class SealTriangularInterferometer(InterferometerList):
    '''SealGW modified bilby TriangularInterferometer.'''

    def __init__(
        self,
        name,
        power_spectral_density,
        minimum_frequency,
        maximum_frequency,
        length,
        latitude,
        longitude,
        elevation,
        xarm_azimuth,
        yarm_azimuth,
        xarm_tilt=0.0,
        yarm_tilt=0.0,
    ):
        super(SealTriangularInterferometer, self).__init__([])
        self.name = name
        # for attr in ['power_spectral_density', 'minimum_frequency', 'maximum_frequency']:
        if isinstance(power_spectral_density, PowerSpectralDensity):
            power_spectral_density = [power_spectral_density] * 3
        if isinstance(minimum_frequency, float) or isinstance(minimum_frequency, int):
            minimum_frequency = [minimum_frequency] * 3
        if isinstance(maximum_frequency, float) or isinstance(maximum_frequency, int):
            maximum_frequency = [maximum_frequency] * 3

        for ii in range(3):
            self.append(
                SealInterferometer(
                    "{}{}".format(name, ii + 1),
                    power_spectral_density[ii],
                    minimum_frequency[ii],
                    maximum_frequency[ii],
                    length,
                    latitude,
                    longitude,
                    elevation,
                    xarm_azimuth,
                    yarm_azimuth,
                    xarm_tilt,
                    yarm_tilt,
                )
            )

            xarm_azimuth += 240
            yarm_azimuth += 240

            latitude += (
                np.arctan(
                    length * np.sin(xarm_azimuth * np.pi / 180) * 1e3 / radius_of_earth
                )
                * 180
                / np.pi
            )
            longitude += (
                np.arctan(
                    length * np.cos(xarm_azimuth * np.pi / 180) * 1e3 / radius_of_earth
                )
                * 180
                / np.pi
            )


def get_empty_Sealinterferometer(name):
    """
    Get an SealGW-modified interferometer with standard parameters for known detectors.

    These objects do not have any noise instantiated.

    The available instruments are:
        H1, L1, V1, GEO600, CE

    Detector positions taken from:
        L1/H1: LIGO-T980044-10
        V1/GEO600: arXiv:gr-qc/0008066 [45]
        CE: located at the site of H1

    Detector sensitivities:
        H1/L1/V1: https://dcc.ligo.org/LIGO-P1200087-v42/public
        GEO600: http://www.geo600.org/1032083/GEO600_Sensitivity_Curves
        CE: https://dcc.ligo.org/LIGO-P1600143/public


    Parameters
    ==========
    name: str
        Interferometer identifier.

    Returns
    =======
    interferometer: SealInterferometer
        SealInterferometer instance
    """
    filename = os.path.join(
        BILBY_ROOT_PATH, "gw", "detector", "detectors", "{}.interferometer".format(name)
    )
    try:
        return load_Sealinterferometer(filename)
    except OSError:
        raise ValueError("Interferometer {} not implemented".format(name))


def load_Sealinterferometer(filename):
    """Load an SealGW-modified interferometer from a file."""
    parameters = dict()
    with open(filename, "r") as parameter_file:
        lines = parameter_file.readlines()
        for line in lines:
            if line[0] == "#" or line[0] == "\n":
                continue
            split_line = line.split("=")
            key = split_line[0].strip()
            value = eval("=".join(split_line[1:]))
            parameters[key] = value
    if "shape" not in parameters.keys():
        ifo = SealInterferometer(**parameters)
        logger.debug("Assuming L shape for {}".format("name"))
    elif parameters["shape"].lower() in ["l", "ligo"]:
        parameters.pop("shape")
        ifo = SealInterferometer(**parameters)
    elif parameters["shape"].lower() in ["triangular", "triangle"]:
        parameters.pop("shape")
        ifo = SealTriangularInterferometer(**parameters)
    else:
        raise IOError(
            "{} could not be loaded. Invalid parameter 'shape'.".format(filename)
        )
    return ifo


class SealInterferometerList(InterferometerList):
    """A list of SealInterferometer objects"""

    def __init__(self, interferometers):
        """Instantiate a InterferometerList

        The InterferometerList is a list of Interferometer objects, each
        object has the data used in evaluating the likelihood

        Parameters
        ==========
        interferometers: iterable
            The list of interferometers
        """

        super(InterferometerList, self).__init__()
        if type(interferometers) == str:
            raise TypeError("Input must not be a string")
        for ifo in interferometers:
            if type(ifo) == str:
                ifo = get_empty_Sealinterferometer(ifo)
            if type(ifo) not in [
                Interferometer,
                TriangularInterferometer,
                SealInterferometer,
                SealTriangularInterferometer,
            ]:
                raise TypeError(
                    "Input list of interferometers are not all Interferometer objects"
                )
            else:
                self.append(ifo)
        self._check_interferometers()

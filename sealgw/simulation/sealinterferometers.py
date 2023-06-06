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

# from ..calculation.localization import lal_et_response_function, lal_ce_response_function, lal_dt_function
from ..calculation.localization import lal_response_function, lal_dt_function
from .generating_data import f_of_tau, tau_of_f, segmentize_tau


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
        antenna_response_change=True,
        antenna_response_change_timescale=8.0,
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
        antenna_response_change: Bool
            whether consider the change in antenna response functions (i.e. the Earth rotation)
        antenna_response_change_timescale:
            time above which antenna_response_change is considered. Only comes to effect when antenna_response_change is True
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
        self.antenna_response_change = antenna_response_change
        self.antenna_response_change_timescale = antenna_response_change_timescale

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
            if self.name in ['ET1', 'ET2', 'ET3', 'CE', 'CEL']:
                return lal_response_function(ra, dec, time, psi, self.name, mode)
            else:
                polarization_tensor = get_polarization_tensor(ra, dec, time, psi, mode)
                return three_by_three_matrix_contraction(
                    self.geometry.detector_tensor, polarization_tensor
                )
        elif mode == self.name:
            return 1
        else:
            return 0

    def time_delay_from_geocenter(self, ra, dec, time):
        """
        Calculate the time delay from the geocenter for the interferometer.

        Use the time delay function from utils.

        Parameters
        ==========
        ra: float
            right ascension of source in radians
        dec: float
            declination of source in radians
        time: float
            GPS time

        Returns
        =======
        float: The time delay from geocenter in seconds
        """
        if self.name in ['ET1', 'ET2', 'ET3', 'CE', 'CEL']:
            return lal_dt_function(ra, dec, time, self.name)
        else:
            # print('!!!!!!')
            return time_delay_from_geocenter(self.geometry.vertex, ra, dec, time)

    def get_detector_response(
        self, waveform_polarizations, parameters, frequencies=None
    ):
        """Get the detector response for a particular waveform

        Parameters
        ==========
        waveform_polarizations: dict
            polarizations of the waveform
        parameters: dict
            parameters describing position and time of arrival of the signal
        frequencies: array-like, optional
        The frequency values to evaluate the response at. If
        not provided, the response is computed using
        :code:`self.frequency_array`. If the frequencies are
        specified, no frequency masking is performed.
        Returns
        =======
        array_like: A 3x3 array representation of the detector response (signal observed in the interferometer)
        """
        if frequencies is None:
            frequencies = self.frequency_array[self.frequency_mask]
            mask = self.frequency_mask
        else:
            mask = np.ones(len(frequencies), dtype=bool)

        # !!Assume only low freqs are masked!!
        # So that masked_index + masked_length = unmasked_index
        masked_length = len(self.frequency_array) - len(frequencies)
        signal = {}

        if self.antenna_response_change:
            try:
                tau = tau_of_f(frequencies, mc=parameters['chirp_mass'])
            except:
                tau = tau_of_f(
                    frequencies, m1=parameters['mass_1'], m2=parameters['mass_2']
                )
            times = parameters['geocent_time'] - tau
            segment_starts = segmentize_tau(tau, self.antenna_response_change_timescale)

            antenna_response_array_dict = dict()
            for mode in waveform_polarizations.keys():
                antenna_response_array_dict[mode] = np.zeros(
                    len(waveform_polarizations[mode])
                )
                for iseg, segment_start in enumerate(segment_starts):
                    time = times[segment_start]
                    if iseg < len(segment_starts) - 1:
                        this_seg_start = segment_start + masked_length
                        next_seg_start = segment_starts[iseg + 1] + masked_length
                        length_to_fill = next_seg_start - this_seg_start
                    else:
                        this_seg_start = segment_start + masked_length
                        next_seg_start = None
                        length_to_fill = (
                            len(waveform_polarizations[mode]) - this_seg_start
                        )
                    antenna_response_array_dict[mode][
                        this_seg_start:next_seg_start
                    ] = np.full(
                        length_to_fill,
                        self.antenna_response(
                            parameters['ra'],
                            parameters['dec'],
                            time,
                            parameters['psi'],
                            mode,
                        ),
                    )

                signal[mode] = (
                    waveform_polarizations[mode] * antenna_response_array_dict[mode]
                )

            signal_ifo = sum(signal.values()) * mask

            time_shift = np.zeros(len(frequencies))
            for iseg, segment_start in enumerate(segment_starts):
                time = times[segment_start]
                if iseg < len(segment_starts) - 1:
                    this_seg_start = segment_start
                    next_seg_start = segment_starts[iseg + 1]
                    length_to_fill = next_seg_start - this_seg_start
                else:
                    this_seg_start = segment_start
                    next_seg_start = None
                    length_to_fill = len(frequencies) - this_seg_start
                time_shift[this_seg_start:next_seg_start] = np.full(
                    length_to_fill,
                    self.time_delay_from_geocenter(
                        parameters['ra'], parameters['dec'], time
                    ),
                )

            # Be careful to first subtract the two GPS times which are ~1e9 sec.
            # And then add the time_shift which varies at ~1e-5 sec
            dt_geocent = (
                parameters['geocent_time'] - self.strain_data.start_time
            )  # not times-?

            dt = dt_geocent + time_shift

        else:
            for mode in waveform_polarizations.keys():
                det_response = self.antenna_response(
                    parameters['ra'],
                    parameters['dec'],
                    parameters['geocent_time'],
                    parameters['psi'],
                    mode,
                )

                signal[mode] = waveform_polarizations[mode] * det_response
            signal_ifo = sum(signal.values()) * mask
            time_shift = self.time_delay_from_geocenter(
                parameters['ra'], parameters['dec'], parameters['geocent_time']
            )

            # Be careful to first subtract the two GPS times which are ~1e9 sec.
            # And then add the time_shift which varies at ~1e-5 sec
            dt_geocent = parameters['geocent_time'] - self.strain_data.start_time
            dt = dt_geocent + time_shift

        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)

        signal_ifo[mask] *= self.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.name), **parameters
        )

        return signal_ifo

    def inject_signal_from_waveform_polarizations(
        self, parameters, injection_polarizations, print_para=True, print_snr=True
    ):
        """Inject a signal into the detector from a dict of waveform polarizations.
        Alternative to `inject_signal` and `inject_signal_from_waveform_generator`.

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`.

        """
        if not self.strain_data.time_within_data(parameters['geocent_time']):
            logger.warning(
                'Injecting signal outside segment, start_time={}, merger time={}.'.format(
                    self.strain_data.start_time, parameters['geocent_time']
                )
            )
        signal_ifo = self.get_detector_response(injection_polarizations, parameters)
        self.strain_data.frequency_domain_strain += signal_ifo

        self.meta_data['optimal_SNR'] = np.sqrt(
            self.optimal_snr_squared(signal=signal_ifo)
        ).real
        self.meta_data['matched_filter_SNR'] = self.matched_filter_snr(
            signal=signal_ifo
        )
        self.meta_data['parameters'] = parameters
        if print_snr:
            logger.info("Injected signal in {}:".format(self.name))
            logger.info("  optimal SNR = {:.2f}".format(self.meta_data['optimal_SNR']))
            logger.info(
                "  matched filter SNR = {:.2f}".format(
                    self.meta_data['matched_filter_SNR']
                )
            )
        if print_para:
            for key in parameters:
                logger.info('  {} = {}'.format(key, parameters[key]))

    def inject_signal(
        self,
        parameters,
        injection_polarizations=None,
        waveform_generator=None,
        raise_error=True,
        print_snr=True,
        print_para=True,
    ):
        """General signal injection method.
        Provide the injection parameters and either the injection polarizations
        or the waveform generator to inject a signal into the detector.
        Defaults to the injection polarizations is both are given.

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict, optional
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`. If
           `waveform_generator` is also given, the injection_polarizations will
           be calculated directly and this argument can be ignored.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator, optional
            A WaveformGenerator instance using the source model to inject. If
            `injection_polarizations` is given, this will be ignored.
        raise_error: bool
            If true, raise an error if the injected signal has a duration
            longer than the data duration. If False, a warning will be printed
            instead.

        Notes
        =====
        if your signal takes a substantial amount of time to generate, or
        you experience buggy behaviour. It is preferable to provide the
        injection_polarizations directly.

        Returns
        =======
        injection_polarizations: dict
            The injected polarizations. This is the same as the injection_polarizations parameters
            if it was passed in. Otherwise it is the return value of waveform_generator.frequency_domain_strain().

        """
        self.check_signal_duration(parameters, raise_error)

        if injection_polarizations is None and waveform_generator is None:
            raise ValueError(
                "inject_signal needs one of waveform_generator or "
                "injection_polarizations."
            )
        elif injection_polarizations is not None:
            self.inject_signal_from_waveform_polarizations(
                parameters=parameters,
                injection_polarizations=injection_polarizations,
                print_snr=print_snr,
                print_para=print_para,
            )
        elif waveform_generator is not None:
            injection_polarizations = self.inject_signal_from_waveform_generator(
                parameters=parameters, waveform_generator=waveform_generator
            )
        return injection_polarizations


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

    if name in ['CEL']:
        filename = os.path.join(
            os.path.dirname(__file__),
            "added_detectors",
            "{}.interferometer".format(name),
        )
    else:
        filename = os.path.join(
            BILBY_ROOT_PATH,
            "gw",
            "detector",
            "detectors",
            "{}.interferometer".format(name),
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

    def inject_signal(
        self,
        parameters=None,
        injection_polarizations=None,
        waveform_generator=None,
        raise_error=True,
        print_para=True,
        print_snr=True,
    ):
        """Inject a signal into noise in each of the three detectors.

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`. If
           `waveform_generator` is also given, the injection_polarizations will
           be calculated directly and this argument can be ignored.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject. If
            `injection_polarizations` is given, this will be ignored.
        raise_error: bool
            Whether to raise an error if the injected signal does not fit in
            the segment.

        Notes
        =====
        if your signal takes a substantial amount of time to generate, or
        you experience buggy behaviour. It is preferable to provide the
        injection_polarizations directly.

        Returns
        =======
        injection_polarizations: dict

        """
        if injection_polarizations is None:
            if waveform_generator is not None:
                injection_polarizations = waveform_generator.frequency_domain_strain(
                    parameters
                )
            else:
                raise ValueError(
                    "inject_signal needs one of waveform_generator or "
                    "injection_polarizations."
                )

        all_injection_polarizations = list()
        for interferometer in self:
            all_injection_polarizations.append(
                interferometer.inject_signal(
                    parameters=parameters,
                    injection_polarizations=injection_polarizations,
                    raise_error=raise_error,
                    print_para=print_para,
                    print_snr=print_snr,
                )
            )

        return all_injection_polarizations

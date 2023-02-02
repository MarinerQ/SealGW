# SEmi-Analitical Localization for Gravitational Waves (SealGW)

A semi-analytical approach for sky localization of gravitational waves, tested on LHV network (see [Phys. Rev. D 104, 104008 (2021)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.104008) or [arXiv:2110.01874](https://arxiv.org/abs/2110.01874) for details). We are trying to employ it in [SPIIR pipeline](https://git.ligo.org/lscsoft/spiir/) in LVK's O4 detection run.

## Installation

- Environments: [IGWN Conda Distribution](https://computing.docs.ligo.org/conda/environments/) is highly recommended

- Dependencies: You need to install [SPIIR](https://github.com/tanghyd/spiir) before installing sealgw.

- Download (git clone) sealgw and go into its directory.

- <tt> pip install . </tt>

- To uninstall:  <tt> pip uninstall sealgw </tt>

## Usage

See <tt> share/examples/ </tt> for usage of sealgw.

## TODO

- Early warning development.

# SEmi-Analitical Localization for Gravitational Waves (SealGW) 

A semi-analytical approach for sky localization of gravitational waves, tested on LHV network (see [Phys. Rev. D 104, 104008 (2021)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.104008) or [arXiv:2110.01874](https://arxiv.org/abs/2110.01874) for details). We are trying to employ it in [SPIIR pipeline](https://git.ligo.org/lscsoft/spiir/) in LVK's O4 detection run.

## Script Structure

### musigma_snr_fitting/:

Fit the $\mu,\sigma$-SNR relation, and get the linear coefficients for prior settings. (See Eq. 25-27 in our paper.) To run the fitting code:

1. You need to put PSD file (1st column for frequency, 2nd for PSD) in <tt>musigma_snr_fitting/psd/{YOUR_PSD_LABEL}/{det_name}_psd.txt</tt>. For example, we want to fit the relation for PSD around GW170817, so we create a folder named 170817 under <tt>musigma_snr_fitting/psd/</tt>, and put <tt>H1.txt, L1.txt, V1.txt</tt> there. Detector name for PSD file should be H1, L1, V1, K1, I1, ET or CE. 

2. Run the python script: python musigma_snr_fitting.py [Number_of_Injections] [Network_Name] [Source_Type] [YOUR_PSD_LABEL] [Number_of_CPUs].

Number_of_Injections: I recommend >30000 for a good fitting.

Network_Name: should be a combination of letters "H", "L", "V", "K", "I", "C", "E". For O3, it should be "LHV". For O4, it should be "LHVK".

Source_Type: BNS or BBH

YOUR_PSD_LABEL: as mentioned before. Can be "O4", "O3", ...

Number_of_CPUs: e.g., 4.

Then we can run it like: <tt> python musigma_snr_fitting.py 30000 LHV BNS 170817 4 </tt>

3. The code will produce 4 output files. <tt> psd/{YOUR_PSD_LABEL}/snr_A_xxx.txt </tt> saves SNR and Aijs, <tt>psd/{YOUR_PSD_LABEL}/abcd_xxx.txt </tt> saves linear coefficients. It has 4 elements, say, a,b,c,d. Then you will have $\mu = a\textrm{SNR} + b,\sigma = c\textrm{SNR} + d$. And 2 figures, showing the linear fitting and bimodal fitting. 


## TODO

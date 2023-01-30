# This script is used to generate data from example3 jupyter notebook
# use "nohup python example3-catalog_test.py" to run it
# It takes ~30min to run on my laptop

from sealgw import seal
import sealgw.calculation as sealcal
import sealgw.simulation as sealsim


seal_O2 = seal.Seal('example_outputs/sealconfig_O2_lhv_BNS.txt')

Nsample = 1000
det_name_list = ['L1', 'H1', 'V1']
source_type = "BNS"
ncpu = 4
save_filename = 'example_outputs/catalog_statistics.txt' 
psd_files = ['example_inputs/L1_psd.txt','example_inputs/H1_psd.txt','example_inputs/V1_psd.txt']

catalog_stats = seal_O2.catalog_test(Nsample, det_name_list, source_type, ncpu, save_filename,duration = 320,
                      use_bilby_psd = False, custom_psd_path = psd_files)


# O4 LHV
ifostr="LHV"
psd_path="H1L1V1-O4_psd-1000001000-21600.xml.gz"
label="LHV_O4ExpectedPSDFixMtot" # label should be suitable for naming files

# O3 LHV
#ifostr="LHV"
#psd_path="H1L1V1-REFERENCE_PSD-1263344418-21600.xml.gz"
#label="LHV_O3PSD"

# O4 LH
ifostr="LH"
psd_path="H1L1V1-O4_psd-1000001000-21600.xml.gz"
label="LH_O4ExpectedPSDFixMtot" # label should be suitable for naming files

#########
out_dir="seal_training_outputs/"

# For test (O2 PSD): high_snr_cutoff = 31, Nsample=10000
# For O3/O4: high_snr_cutoff = 35, Nsample=30000
python fit_sealgw_model.py \
    ${ifostr} \
    ${psd_path} \
    ${out_dir} \
    ${label} \
    --low-snr-cutoff 9 \
    --high-snr-cutoff 35 \
    --nsample 30000\
    --ncpu 6
### ET2CE ###
ifostr="ET_CE_CEL"
psd_path="iwanttosleep"
label="ET2CE_BNSEW" 
out_dir="et2ce_training_outputs/"

export OMP_NUM_THREADS=20
python fit_sealgw_model.py \
    ${ifostr} \
    ${psd_path} \
    ${out_dir} \
    ${label} \
    --low-snr-cutoff 9 \
    --high-snr-cutoff 35 \
    --nsample 10000\
    --ncpu 8


### ETCE ###
ifostr="ET_CE"
psd_path="ireallywanttosleep"
label="ETCE_BNSEW" 
out_dir="etce_training_outputs/"

python fit_sealgw_model.py \
    ${ifostr} \
    ${psd_path} \
    ${out_dir} \
    ${label} \
    --low-snr-cutoff 9 \
    --high-snr-cutoff 35 \
    --nsample 10000\
    --ncpu 8
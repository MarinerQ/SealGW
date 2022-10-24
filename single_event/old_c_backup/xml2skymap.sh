# process xml file to txt
module load python/3.10.4 gsl/2.5
source /fred/oz016/qian/spiirenv/bin/activate
python process_xml.py 

# run C code
#source bashrc_c
module load gcc/8.2.0 openmpi/4.0.0 lalsuite/6.49.0 openblas/0.2.20 gsl/2.5 fftw/3.3.8
cp -r /fred/oz016/qian/paper_170817/sky_grids /fred/oz016/qian/loc4spiir/qian-skymap
mkdir skymap

make clean
make
./exe_loc

# plot skymap
#source bashrc_py
module load python/3.10.4
python plot_skymap.py


# delete large files that can not be uploaded to github
#rm -r skymap sky_grids
source bashrc_c
#cp -r /fred/oz016/qian/paper_170817/sky_grids /fred/oz016/qian/loc4spiir/qian-skymap
make clean
make
./exe_170817

source bashrc_py
python plot_skymap.py

#rm -r skymap sky_grids

# Targets:
# 1. Read .xml file
# 2. Store SNR as txt file
# 3. Calculate horizon distance and store
# The process may be faster if we do the above in C code. 

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import spiir

data_dir = Path("/fred/oz016/qian/loc4spiir/qian-skymap/")
coinc_xml = data_dir / "H1L1V1_1187008603_3_606.xml"

snrs = spiir.io.ligolw.array.load_snr_series_from_xml(coinc_xml)
snrs = pd.concat(snrs, axis=1)
snr_real = np.real(snr)
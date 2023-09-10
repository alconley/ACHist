from histogrammer import Histogrammer
import polars as pl
import matplotlib.pyplot as plt 

# read in the data
df = pl.read_parquet("./52Cr_July2023_REU_CeBrA/analysis/run_83_112_gainmatched.parquet")

# create an instance of the Histogrammer class
h = Histogrammer()

# create a 1D histogram
h.histo1d(xdata=df["Xavg"], bins=600, range=(-300,300))

plt.show()

import sys
sys.path.append('/')
from histogrammer import Histogrammer
import polars as pl
import matplotlib.pyplot as plt 

# read in the data
df = pl.read_parquet("52Cr_July2023_REU_CeBrA/built/run_83.parquet")


# print(df.columns) # print the availiable columns

h = Histogrammer() # create an instance of the Histogrammer class

# plot a subplot with the focal plane spectrum and the PID without a cut
fig_NoCut, ax_NoCut = plt.subplots(2,1)
h.histo2d(data=[ (df["ScintLeftEnergy"],df["AnodeBackEnergy"]) ], bins=[512,512], range=[ [0,2048], [0,2048]], subplots=(fig_NoCut, ax_NoCut[0]), cbar=False) #Particle Identification Plot
h.histo1d(xdata=df["Xavg"], bins=600, range=(-300,300), subplots=(fig_NoCut, ax_NoCut[1])) # Spectrum without cut

# filter the data frame with the proton cut created with the 2D histogram example
PID_df = h.filter_df_with_cut(df=df, XColumn="ScintLeftEnergy", YColumn="AnodeBackEnergy", CutFile="./histogrammer/example_cut.json")

# plot a subplot with the focal plane spectrum and the PID with a cut
fig_Cut, ax_Cut = plt.subplots(2,1)
h.histo2d(data=[ (PID_df["ScintLeftEnergy"],PID_df["AnodeBackEnergy"]) ], bins=[512,512], range=[ [0,2048], [0,2048]], subplots=(fig_Cut, ax_Cut[0]), cbar=False) #Particle Identification Plot
h.histo1d(xdata=PID_df["Xavg"], bins=600, range=(-300,300), subplots=(fig_Cut, ax_Cut[1])) # Spectrum without cut

# Create a new column
xavg_ECal_Para = [-0.0023904378617156377,-18.49776562220117, 1457.4874219091237]

a, b, c = xavg_ECal_Para
df = df.with_columns( ( (a*pl.col(f"Xavg")*pl.col(f"Xavg")) + (b*pl.col(f"Xavg")) + (c) ).alias(f"XavgEnergyCalibrated"))


plt.show()

from histogrammer import Histogrammer

import polars as pl
import numpy as np
import matplotlib.pyplot as plt 

df = pl.read_parquet("../52Cr_July2023_REU_CeBrA/analysis/run_83_112_gainmatched.parquet")

#52Cr(d,p)53Cr, 8.3 kG field 
xavg_ECal_Para = [-0.0023904378617156377,-18.49776562220117, 1457.4874219091237]

a, b, c = xavg_ECal_Para
df = df.with_columns( ( (a*pl.col(f"Xavg")*pl.col(f"Xavg")) + (b*pl.col(f"Xavg")) + (c) ).alias(f"XavgEnergyCalibrated"))


# Protons coincidence of the 52Cr at the 8.3 kG field (run 82 through 112)
cebraTimeGate = [ [0,-1158,-1153],
                  [1,-1157,-1151],
                  [2,-1157,-1151],
                  [3,-1156,-1150],
                  [4,-1126,-1120] ]

ECal_cebraE0_Para = [0, 1.7551059351549314, -12.273506897222896]
ECal_cebraE1_Para = [0, 1.9510278378962256, -16.0245754973971]
ECal_cebraE2_Para = [0, 1.917190081718234, 16.430212777833802]
ECal_cebraE3_Para = [0, 1.6931918955746692, 12.021258506937766]
ECal_cebraE4_Para = [0, 1.6373533248536343, 13.091030061910748]

CeBrA_ECal = [ECal_cebraE0_Para, ECal_cebraE1_Para, ECal_cebraE2_Para, ECal_cebraE3_Para, ECal_cebraE4_Para]

det_dfs = []
NumDetectors = 5

pg_data = []
for det in range(NumDetectors):
    
    a,b,c = CeBrA_ECal[det]
    df = df.with_columns( ( a*pl.col(f"Cebra{det}Energy")*pl.col(f"Cebra{det}Energy") + b*pl.col(f"Cebra{det}Energy") + c).alias(f"Cebra{det}EnergyCalibrated"))
    
    det_df = df.filter( pl.col(f"Cebra{det}TimetoScint") > cebraTimeGate[det][1] ).filter( pl.col(f"Cebra{det}TimetoScint") < cebraTimeGate[det][2] )
    
    det_dfs.append(det_df)

    pg_data.append( (det_df["XavgEnergyCalibrated"], det_df[f"Cebra{det}EnergyCalibrated"]) )




h = Histogrammer()

fig, ax = plt.subplots(2,2, figsize=(15,10))
ax = ax.flatten()

det = 3

h.histo1d(xdata=df[f"Cebra{det}TimetoScint"], bins=100, range=(-1200,-1100), subplots=(fig, ax[0]), xlabel=f"Cebra{det}TimetoScint", ylabel="Counts", display_stats=False, color='k')
h.histo1d(xdata=det_dfs[det][f"Cebra{det}TimetoScint"], bins=100, range=(-1200,-1100), subplots=(fig,ax[0]), xlabel=f"Cebra{det}TimetoScint", ylabel="Counts", display_stats=True, color='red')
ax[0].axvline(x=cebraTimeGate[det][1], color='red', linestyle='--', linewidth=0.5, label="Coincidence gate")
ax[0].axvline(x=cebraTimeGate[det][2], color='red', linestyle='--', linewidth=0.5)
ax[0].legend(loc="upper left")

h.histo2d(data=[pg_data[det]], bins=[400,300], range=[[0,5200], [0,5200]], subplots=(fig,ax[1]), xlabel=r"$^{53}$Cr Excitation Energy [keV]",ylabel=r"$\gamma$-ray Energy [keV]")


time_gate_width = cebraTimeGate[det][2] - cebraTimeGate[det][1]

low_gate = cebraTimeGate[det][1] - time_gate_width
high_gate = cebraTimeGate[det][1]

det_df_background_df = df.filter( pl.col(f"Cebra{det}TimetoScint") > low_gate).filter( pl.col(f"Cebra{det}TimetoScint") < high_gate)

h.histo1d(xdata=df[f"Cebra{det}TimetoScint"], bins=100, range=(-1200,-1100), subplots=(fig, ax[2]), xlabel=f"Cebra{det}TimetoScint", ylabel="Counts", display_stats=False, color='k')
h.histo1d(xdata=det_df_background_df[f"Cebra{det}TimetoScint"], bins=100, range=(-1200,-1100), subplots=(fig,ax[2]), xlabel=f"Cebra{det}TimetoScint", ylabel="Counts", display_stats=True, color='blue')
ax[2].axvline(x=low_gate, color='blue', linestyle='--', linewidth=0.5, label="Background coincidence gate")
ax[2].axvline(x=high_gate, color='blue', linestyle='--', linewidth=0.5)

h.histo2d(data=[(det_df_background_df["XavgEnergyCalibrated"], det_df_background_df[f"Cebra{det}EnergyCalibrated"])], bins=[400,300], range=[[0,5200], [0,5200]], subplots=(fig,ax[3]), xlabel=r"$^{53}$Cr Energy [keV]",ylabel=r"$\gamma$-ray Energy [keV]")


plt.show()
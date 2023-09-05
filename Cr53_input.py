from histogrammer import Histogrammer

import polars as pl
import numpy as np
import matplotlib.pyplot as plt 

df = pl.read_parquet("./52Cr_July2023_REU_CeBrA/analysis/run_83_112_gainmatched.parquet")

#52Cr(d,p)53Cr, 8.3 kG field 
xavg_ECal_Para = [-0.0023904378617156377,-18.49776562220117, 1457.4874219091237]

a, b, c = xavg_ECal_Para
df = df.with_columns( ( (a*pl.col(f"Xavg")*pl.col(f"Xavg")) + (b*pl.col(f"Xavg")) + (c) ).alias(f"XavgEnergyCalibrated"))


# Protons coincidence of the 52Cr at the 8.3 kG field (run 82 through 112)
cebraTimeGate = [ [0,-1158,-1152],
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
tcut_xavg = []
for det in range(NumDetectors):
    
    a,b,c = CeBrA_ECal[det]
    df = df.with_columns( ( a*pl.col(f"Cebra{det}Energy")*pl.col(f"Cebra{det}Energy") + b*pl.col(f"Cebra{det}Energy") + c).alias(f"Cebra{det}EnergyCalibrated"))
    
    det_df = df.filter( pl.col(f"Cebra{det}TimetoScint") > cebraTimeGate[det][1] ).filter( pl.col(f"Cebra{det}TimetoScint") < cebraTimeGate[det][2] )
    
    det_dfs.append(det_df)

    pg_data.append( (det_df["XavgEnergyCalibrated"], det_df[f"Cebra{det}EnergyCalibrated"]) )
    tcut_xavg.append(det_df["XavgEnergyCalibrated"])



h = Histogrammer()

# timeToScintFig, timeToScintax = plt.subplots(2,3, figsize=(15,10))
# timeToScintax = timeToScintax.fwlatten()

# for det in range(NumDetectors):
#     h.histo1d(xdata=df[f"Cebra{det}TimetoScint"], bins=100, range=(-1200,-1100), subplots=(timeToScintFig,timeToScintax[det]), xlabel=f"Cebra{det}TimetoScint", ylabel="Counts", display_stats=False, color='k')
#     h.histo1d(xdata=det_dfs[det][f"Cebra{det}TimetoScint"], bins=100, range=(-1200,-1100), subplots=(timeToScintFig,timeToScintax[det]), xlabel=f"Cebra{det}TimetoScint", ylabel="Counts", display_stats=False, color='red')

# plt.show()

# h.histo1d(xdata=df["XavgEnergyCalibrated"], bins=500, range=(0,6000))

# h.histo1d(xdata=[df["Cebra0Energy"]], bins=512, range=(0,4096))

# h.histo2d(data=[ (df["ScintLeftEnergy"],df["AnodeBackEnergy"])], bins=[512,512], range=[ [0,2048], [0,2048]])

# for i in range(5):
#     h.histo2d(data=[[det_dfs[i]["Xavg"],det_dfs[i][f"Cebra{i}EnergyCalibrated"]]] , bins=[600,250], range=[[-300,300], [0,6000]], xlabel="Xavg", ylabel=f"Cebra{i}Energy")

fig, axs = plt.subplots(figsize=(12, 6))
h.histo1d(xdata=df["Xavg"], bins=600, range=(-300,300), subplots=(fig, axs))

# h.histo1d(xdata=df["XavgEnergyCalibrated"], bins=500, range=(0,6000), subplots=(fig, axs))


# h.histo1d(xdata=tcut_xavg, bins=300, range=(0,6000), subplots=(fig, axs))

# h.histo1d(xdata=[df["Cebra0Energy"]], bins=512, range=(0,4096), subplots=(fig, axs))



# ''' 52Cr(d,p)53Cr, 8.3 kG field particle-gamma matrix
# fig, axs = plt.subplots(figsize=(2.8919330289193304, 2.8919330289193304))

# fig, axs = plt.subplots(figsize=(6, 6))

# h.histo2d(data=pg_data, bins=[450,550], range=[[0,5500], [0,5500]], subplots=(fig,axs), xlabel=r"$^{53}$Cr Excitation Energy [keV]",ylabel=r"$\gamma$-ray Energy [keV]",display_stats=False, cbar=False)

# # x = np.linspace(0,5500,5500)
# # gs = x 
# # first_excited = x - 564
# # second_excited = x - 1006
# # third_excited = x - 1289

# # axs.plot(x,gs, color='#17a657',             linewidth=0.5, label=r'$\gamma$ decay to $\frac{3}{2}^{-}$', alpha=0.8, linestyle='-')
# # axs.plot(x,first_excited, color='#751a9c',  linewidth=0.5, label=r'$\gamma$ decay to $\frac{1}{2}^{-}$', alpha=0.8, linestyle='--')
# # axs.plot(x,second_excited, color='#a61753', linewidth=0.5, label=r'$\gamma$ decay to $\frac{5}{2}^{-}$', alpha=0.8, linestyle='-.')
# # axs.plot(x,third_excited, color='#de8407',  linewidth=1, label=r'$\gamma$ decay to $\frac{7}{2}^{-}$', alpha=0.8, linestyle=':')

# # offset = 200
# # axs.text(564,  564+offset,  r"564 keV (J$^{\pi}$=$\frac{1}{2}^{-}$)", rotation=90, verticalalignment='bottom', horizontalalignment='center',bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1,  ec="none" ))
# # axs.text(1006,1006+offset,r"1006 keV (J$^{\pi}$=$\frac{5}{2}^{-}$)", rotation=90, verticalalignment='bottom', horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1,  ec="none" ))
# # axs.text(1300,1300+offset,r"1289 keV (J$^{\pi}$=$\frac{7}{2}^{-}$)", rotation=90, verticalalignment='bottom', horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1,  ec="none" ))
# # axs.text(2320,2320+offset,r"2320 keV (J$^{\pi}$=$\frac{3}{2}^{-}$)", rotation=90, verticalalignment='bottom', horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1,  ec="none" ))
# # axs.text(3706,3706+offset,r"3707 keV (J$^{\pi}$=$\frac{9}{2}^{+}$)", rotation=90, verticalalignment='bottom', horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1,  ec="none" ))
# # axs.vlines(3700,2600,3706+offset-50, color='#1cadba', linewidth=0.5, alpha=1, linestyle='--')
# # axs.legend(loc='upper left',shadow=False, frameon=True, fancybox=False, edgecolor='none', facecolor='none')

# fig.subplots_adjust(top=0.995,
# bottom=0.112,
# left=0.147,
# right=0.988,
# hspace=0.2,
# wspace=0.2)

# axs.minorticks_on()
# axs.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
# axs.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)


# plt.savefig(f"./histogrammer/53Cr_pg_matrix_labeled.pdf",format='pdf')
# plt.savefig(f"./histogrammer/53Cr_pg_matrix_labeled.png",format='png',dpi=1200)


# '''


plt.show()

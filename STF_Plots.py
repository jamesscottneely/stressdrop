import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d.proj3d import transform

import stress_func as sf


k = 0.37 # Shape factor brune calculation
c = 1 # Constant for Fc to T conversion

beta = 3600. # s-wave velocity m/s

def str2Array(params):
    strVal = params.values[0]
    params_array = [float(vals) for vals in strVal[1:-1].split(',')]
    return params_array

EQ_ID = '202509301759246336_2983'#### Enter the EQ id for plotting
eqdirs = 'Brune_0.50SD_100HZ_Uniform'
path = "/Users/jamesneely/Documents/NSF/StressDrop_Bands/" +eqdirs
maxHZ_i = 5
### Read in STF file
STF_data = pd.read_csv(path+'/STFs/' +EQ_ID+ ".txt")

### Read in EQ file with data
eqData = pd.read_csv(path+'/eqFile.txt',sep='|')
eqData = eqData[eqData.eq_id==EQ_ID]
print(eqData.iloc[0])
### Read in fitting data
fitData = pd.read_csv(path+'/FittingFile.csv')
fitData = fitData[fitData.eq_id==EQ_ID]
print(fitData.iloc[0])
### Read in EGF File
egfData = pd.read_csv(path+ '/EGF_1MPa/EGF_1MPa.txt')
egfData = egfData[egfData.MAIN_ID==EQ_ID]
##### Read in STF file
STFData = pd.read_csv(path + "/STFs/" +EQ_ID +".txt")
print(STFData)

###### Plot STF information
fig,ax = plt.subplots(nrows=1,ncols=2,layout='constrained',figsize=(8,4))
#### Plot time series
print(eqData.t_i.values)
# print(eqData.t_i.values[0])
# Plot main STF
ax[0].plot(STFData.time,STFData.STF,lw=3)
####### Plot individual pulses
# for pulse in range(eqData.num_events):
#     pulse_sub = sf.brune_time_moment(np.atleast_2d(moment_sub).T, np.atleast_2d(fc_sub).T,
#                                      np.atleast_2d(start_sub).T + np.mean(t), np.tile(t, (int(num_events),
#                                                                                           1)))  # +50 on the start_sub ensures that it is shifted properly away from 0. Need to change shape for calculation
#     pulse_sub[pulse_sub < 0] = 0  # Remove negative values
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Moment Rate (N-m/s)')
xmin = 500 + np.sum(str2Array(eqData.t_i)) -10
xmax = 500 + np.max(str2Array(eqData.t_i)) +10
ax[0].set_xlim([xmin,xmax])
textStr = r"$M_w$: {:.1f}".format(eqData.mw_tot.iloc[0]) +"\n"+ r"$\Delta\sigma_{{Area}}$: {:.1f}MPa".format(eqData.sigma_tot_area.iloc[0]) +"\n"+ r"$\Delta\sigma_{{Mo}}$: {:.1f}MPa".format(eqData.sigma_tot_moment.iloc[0])
ax[0].text(0,.99,textStr,ha='left',va='top',transform=ax[0].transAxes)
######
###### Plot the spectral amplitude
######
freq,amplitude,FT_complex = sf.sig_process(STFData.time.to_numpy(),STFData.STF.to_numpy(),[0,maxHZ_i])
ax[1].loglog(freq,amplitude)
## Plot best fitting line
bestBrune = sf.bruneMod(freq,[amplitude[0],fitData.iloc[0].Fc_2_Mo])
specstr = r'$f_c$: {:.2f}hz'.format(fitData.iloc[0].Fc_2_Mo) + "\n" + r'$\Delta\sigma$: {:.1f}MPa'.format(fitData.iloc[0].SIG_2_Mo)
### TRim plot
freq,amplitude,FT_complex = sf.sig_process(STFData.time.to_numpy(),STFData.STF.to_numpy(),[0,maxHZ_i])
bestBrune = sf.bruneMod(freq,[amplitude[0],fitData.iloc[0].Fc_2_Mo])
ax[1].loglog(freq,bestBrune)
ax[1].text(.99,.99,specstr,ha='right',va='top',transform=ax[1].transAxes)
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Moment (N-m)")


##### Save figure
outFile= path + '/Figures/STF_' +EQ_ID +".png"
fig.savefig(outFile,dpi=500)


######
###### Plot the EGF curves
######
log_samp = 0.025
###### Plot STF information
fig,ax = plt.subplots(nrows=1,ncols=2,layout='constrained',figsize=(8,4))

egfData = egfData[egfData.MaxHz==maxHZ_i]
mergeData = eqData.merge(egfData,how='inner',left_on='eq_id',right_on='MAIN_ID')
mag_dif = 1.5
mergeData = mergeData[(mergeData.eq_id==EQ_ID)]
print(mergeData)
######
###### Plot the spectral amplitude
######
ax[0].loglog(freq,amplitude)
### PLot Ratio 1
fc_egf_mo = sf.mw2mo(mergeData['mw_tot'][0]-mag_dif)
fc_egf = sf.stress2fc(1, fc_egf_mo, k, beta)
print(fc_egf_mo,fc_egf)
sim_brune= sf.bruneMod(freq,[fc_egf_mo,fc_egf])
ax[0].loglog(freq,sim_brune,color='orange')
# mergeData.Fc_2_Mo_EGF[0]]
### PLot Ratio 2
fc_egf_mo_big = sf.mw2mo(mergeData['mw_tot'][0]+mag_dif)
fc_egf_big = sf.stress2fc(1, fc_egf_mo_big, k, beta)
sim_brune_big= sf.bruneMod(freq,[fc_egf_mo_big,fc_egf_big])
ax[0].loglog(freq,sim_brune_big,color='green')
# ax[0].loglog(freq,amplitude/sim_brune)
# ax[0].text(.99,.99,specstr,ha='right',va='top',transform=ax[0].transAxes)
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Moment (N-m)")
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Moment (N-m)")

##### Spectral ratio 1
ratio = amplitude[1:]/sim_brune[1:]
freq_trim = freq[1:]
freq_main_smooth, ratio_main_smooth = sf.resampSpec(freq_trim,ratio, log_samp)
ax[1].loglog(freq_main_smooth, ratio_main_smooth,color='orange',alpha=.75)
## plot best fitting egf with fixed Mo
fc_main = mergeData.Fc_2_Mo_Main_comp_simp[0]
sig_main = mergeData.SIG_2_Mo_Main_comp_simp[0]
fc_egf = mergeData.Fc_2_Mo_EGF_comp_simp[0]
sig_egf = mergeData.SIG_2_Mo_EGF_comp_simp[0]
bestFit = sf.ratio_bruneModMoFixed(freq_main_smooth,[fc_main,fc_egf],mergeData['moment_tot'][0],fc_egf_mo)
ax[1].loglog(freq_main_smooth, bestFit,color='orange')
## plot best fitting egf with free Mo
fc_main = mergeData.Fc_2_MoFree_Main_comp_simp[0]
sig_main = mergeData.SIG_2_MoFree_Main_comp_simp[0]
fc_egf = mergeData.Fc_2_MoFree_EGF_comp_simp[0]
sig_egf = mergeData.SIG_2_MoFree_EGF_comp_simp[0]
Mo_free = mergeData.MoFree_comp_simp[0]
bestFit = sf.ratio_bruneModMoFree(freq_main_smooth,[fc_main,fc_egf,Mo_free])
ax[1].loglog(freq_main_smooth, bestFit,color='orange',ls=':')

##### Spectral ratio 2
ratio = sim_brune_big[1:]/amplitude[1:]
freq_trim = freq[1:]
freq_main_smooth, ratio_main_smooth = sf.resampSpec(freq_trim,ratio, log_samp)
ax[1].loglog(freq_main_smooth, ratio_main_smooth,color='green',alpha=.5)
## plot best fitting egf
fc_main = mergeData.Fc_2_Mo_Main_simp_comp[0]
sig_main = mergeData.SIG_2_Mo_Main_simp_comp[0]
fc_egf = mergeData.Fc_2_Mo_EGF_simp_comp[0]
sig_egf = mergeData.SIG_2_Mo_EGF_simp_comp[0]
bestFit = sf.ratio_bruneModMoFixed(freq_main_smooth,[fc_main,fc_egf],fc_egf_mo_big,mergeData['moment_tot'][0])
ax[1].loglog(freq_main_smooth, bestFit,color='green')
## plot best fitting egf with free Mo
fc_main = mergeData.Fc_2_MoFree_Main_simp_comp[0]
sig_main = mergeData.SIG_2_MoFree_Main_simp_comp[0]
fc_egf = mergeData.Fc_2_MoFree_EGF_simp_comp[0]
sig_egf = mergeData.SIG_2_MoFree_EGF_simp_comp[0]
Mo_free = mergeData.MoFree_simp_comp[0]
bestFit = sf.ratio_bruneModMoFree(freq_main_smooth,[fc_main,fc_egf,Mo_free])
ax[1].loglog(freq_main_smooth, bestFit,color='green',ls=":")

## pull egf data
# print(egfData.iloc[0])
# egfData_trim = egfData[egfData.MaxHz == 25]




# ax[1].text(.99,.99,specstr,ha='right',va='top',transform=ax[0].transAxes)

ax[1].set_ylabel('Spectral Ratio')
ax[1].set_xlabel('Frequency (Hz)')



outFile= path + '/Figures/EGF_' +EQ_ID +".png"
fig.savefig(outFile,dpi=500)

# print(egfData.iloc[0]) - Rerun and rename spec ratio files for STFs
# plot the spectral ratio then plot the best fitting curves
# bruneMain = sf.bruneMod(freq,[amplitude[0],fitData.iloc[0].Fc_2_Mo])



######################################

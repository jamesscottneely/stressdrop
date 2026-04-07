##### Plot to look for magnitude/stress drop trends
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import matplotlib.patheffects as pe


def median_bin(xvals, yvals, ax):
    xvals = xvals.to_numpy()
    yvals = yvals.to_numpy()
    shiftval = .25
    intval = shiftval / 2

    m_pts = np.arange(5.5, 8.25, shiftval)
    for m in m_pts:
        # print(m)
        min_idx = xvals >= (m - intval)
        max_idx = xvals < (m + intval)
        idx_true = [idx for idx in range(len(xvals)) if min_idx[idx] and max_idx[idx]]  # Get ones in range
        indata = (yvals[idx_true],)
        median_bin = np.median(indata)
        try:
            res = scipy.stats.bootstrap(indata, np.median, n_resamples=1000, confidence_level=0.95)
            asymmetric_error = np.array(
                [[median_bin - res.confidence_interval[0], res.confidence_interval[1] - median_bin]]).T
            ax.errorbar(m, median_bin, yerr=asymmetric_error, color='black', fmt='o', markeredgecolor='black', capsize=5,
                        markerfacecolor='yellow')
            ax.text(m, median_bin + median_bin * 0.1, "{:.1f}".format(median_bin), va='bottom', ha='center',
                        color='black', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        except:
            continue


###### DataPath
dataPath = "/Users/jamesneely/Documents/NSF/StressDrop_Bands/Brune_0.50SD_100HZ"
### Read Data
truth = pd.read_csv(dataPath + "/eqFile.txt",sep='|')
# estimate = pd.read_csv(dataPath + "/FittingFile.csv")
# ### Merge files
# merge_df = truth.merge(estimate, how='inner',on='eq_id')

##### Pull EGFs
EGFs = pd.read_csv(dataPath +"/EGFs/EGFs.txt")
finalMerge = pd.merge(truth,EGFs,right_on=['MAIN_ID'],left_on=['eq_id'],how='left',suffixes=('_Truth','_RatioFit'))

print(finalMerge.iloc[0])
print(finalMerge)


####################
#### Create plot type 1
####################
fig,ax = plt.subplots(nrows=5,ncols=1,figsize=(8,16),sharex=True)
### Plot the truth
HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
### Loop through band plots
for idx in np.arange(0,5,1):
    print(idx)
    merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
    grouped_df = merge_dfTEMP.groupby(['MAIN_ID'])
    # print(merge_dfTEMP.iloc[0])
    # print(merge_dfTEMP)
    #### Fix n=2
    ax[idx].semilogy(grouped_df.mw_tot.mean(), grouped_df.SIG_2_Mo_Main.apply(scipy.stats.gmean),'o', alpha=.1)
    median_bin(grouped_df.mw_tot.mean(), grouped_df.SIG_2_Mo_Main.apply(scipy.stats.gmean), ax[idx])
    ax[idx].set_xlabel('Mw')
    ax[idx].set_ylabel(r'$\Delta\sigma$ (Fc n=2)')
    ax[idx].set_title("Spec Ratio Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))

ax[0].set_ylim([5e-3,5e1])
fig.tight_layout()
fig.savefig(dataPath+'/EGFs/Figures/'+'StressDropTrends.png')
#
####################
##### Plot 2 - Mean estmimate to true value
####################
fig,ax = plt.subplots(nrows=5,ncols=1,figsize=(8,16),sharex=True)
### Plot the truth
HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
### Loop through band plots
for idx in np.arange(0,5,1):
    print(idx)
    merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
    grouped_df = merge_dfTEMP.groupby(['MAIN_ID'])
    # print(merge_dfTEMP.iloc[0])
    # print(merge_dfTEMP)
    #### Fix n=2
    ax[idx].semilogy(grouped_df.mw_tot.mean(), grouped_df.SIG_2_Mo_Main.apply(scipy.stats.gmean)/grouped_df.sigma_tot_moment.mean(),'o', alpha=.1)
    median_bin(grouped_df.mw_tot.mean(), grouped_df.SIG_2_Mo_Main.apply(scipy.stats.gmean)/grouped_df.sigma_tot_moment.mean(), ax[idx])
    ax[idx].set_xlabel('Mw')
    ax[idx].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Mo (Fc n=2)')
    ax[idx].set_title("Spec Ratio Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))

ax[0].set_ylim([5e-3,5e1])
fig.tight_layout()
fig.savefig(dataPath+'/EGFs/Figures/'+'Ratio_Truth_MoWeight.png')
#
####################
##### Plot 3 - Mean estmimate to true value (Area)
####################
fig,ax = plt.subplots(nrows=5,ncols=1,figsize=(8,16),sharex=True)
### Plot the truth
HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
### Loop through band plots
for idx in np.arange(0,5,1):
    print(idx)
    merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
    grouped_df = merge_dfTEMP.groupby(['MAIN_ID'])
    # print(merge_dfTEMP.iloc[0])
    # print(merge_dfTEMP)
    #### Fix n=2
    ax[idx].semilogy(grouped_df.mw_tot.mean(), grouped_df.SIG_2_Mo_Main.apply(scipy.stats.gmean)/grouped_df.sigma_tot_area.mean(),'o', alpha=.1)
    median_bin(grouped_df.mw_tot.mean(), grouped_df.SIG_2_Mo_Main.apply(scipy.stats.gmean)/grouped_df.sigma_tot_area.mean(), ax[idx])
    ax[idx].set_xlabel('Mw')
    ax[idx].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Area (Fc n=2)')
    ax[idx].set_title("Spec Ratio Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))

ax[0].set_ylim([5e-3,5e1])
fig.tight_layout()
fig.savefig(dataPath+'/EGFs/Figures/'+'Ratio_Truth_AreaWeight.png')


####################
##### Plot 4 - Main estimate to ratio estimate
####################
fig,ax = plt.subplots(nrows=5,ncols=1,figsize=(8,16),sharex=True)
### Plot the truth
HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
### Loop through band plots
for idx in np.arange(0,5,1):
    print(idx)
    merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
    grouped_df_MAIN = merge_dfTEMP.groupby(['MAIN_ID']).SIG_2_Mo_Main.apply(scipy.stats.gmean).to_frame()
    grouped_df_EGF = merge_dfTEMP.groupby(['EGF_ID']).SIG_2_Mo_EGF.apply(scipy.stats.gmean).to_frame()
    group_final = grouped_df_MAIN.merge(grouped_df_EGF,how='inner',left_on='MAIN_ID',right_on='EGF_ID')
    print(group_final)
    # print(merge_dfTEMP.iloc[0])
    # print(merge_dfTEMP)
    #### Fix n=2
    ax[idx].loglog(group_final.SIG_2_Mo_Main, group_final.SIG_2_Mo_EGF,'o', alpha=.1)
    ax[idx].set_xlabel(r'$\Delta\sigma$MAIN')
    ax[idx].set_ylabel(r'/$\Delta\sigma$EGF')
    ax[idx].set_title("Spec Ratio Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
    ax[idx].plot([1e-2,1e2],[1e-2,1e2],ls='--',color='grey')
fig.tight_layout()
fig.savefig(dataPath+'/EGFs/Figures/'+'Ratio_Main_EGF.png')

####################
##### Plot 4 - Main estimate to ratio estimate - histogram
####################
fig,ax = plt.subplots(nrows=5,ncols=1,figsize=(8,16),sharex=True)
### Plot the truth
HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
### Loop through band plots
for idx in np.arange(0,5,1):
    print(idx)
    merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
    grouped_df_MAIN = merge_dfTEMP.groupby(['MAIN_ID']).SIG_2_Mo_Main.apply(scipy.stats.gmean).to_frame()
    grouped_df_EGF = merge_dfTEMP.groupby(['EGF_ID']).SIG_2_Mo_EGF.apply(scipy.stats.gmean).to_frame()
    group_final = grouped_df_MAIN.merge(grouped_df_EGF,how='inner',left_on='MAIN_ID',right_on='EGF_ID')
    print(group_final)
    # print(merge_dfTEMP.iloc[0])
    # print(merge_dfTEMP)
    #### Fix n=2
    median = np.median(np.log10(group_final.SIG_2_Mo_Main / group_final.SIG_2_Mo_EGF))
    ax[idx].hist(np.log10(group_final.SIG_2_Mo_Main/group_final.SIG_2_Mo_EGF),label="Med {:.1f}".format(median))
    ax[idx].set_xlabel(r'$\Delta\sigma$MAIN/$\Delta\sigma$EGF')
    ax[idx].set_ylabel('Count')
    ax[idx].axvline(0,ls='--',color='red')
    ax[idx].set_title("Spec Ratio Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
    ax[idx].legend()
fig.tight_layout()
fig.savefig(dataPath+'/EGFs/Figures/'+'Ratio_Main_EGF_HIST.png')

####################
#### Create plot type 2
####################
# fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(10,20),sharey=True)
# ### Plot the truth
# # HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
# # HZ_Bands = [[.001,50],[.01,25],[.1,10],[.1,5],[.5,2]]
#
# ### Loop through band plots
# for idx in np.arange(0,5,1):
#     merge_dfTEMP = merge_df[(merge_df.MinHZ==HZ_Bands[idx][0]) & (merge_df.MaxHz==HZ_Bands[idx][1])]
#     #### Fix n=2
#     ax[idx,0].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_area,'o', alpha=.1)
#     median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_area, ax[idx,0])
#     ax[idx,0].set_xlabel('Mw')
#     ax[idx,0].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Area (Fc n=2)')
#     ax[idx,0].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#     #### n=?
#     ax[idx,1].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_area, 'o', alpha=.1)
#     median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_area, ax[idx,1])
#     ax[idx,1].set_xlabel('Mw')
#     ax[idx,1].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Area  (Fc n=?)')
#     ax[idx,1].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#
# ax[0,0].set_ylim([5e-3,5e1])
# fig.tight_layout()
# fig.savefig(dataPath+'Figures/'+'StressDropTrends_Ratio_Area.png')
#
# ####################
# #### Create plot type 3
# ####################
# fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(10,20),sharey=True)
# ### Plot the truth
# # HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
# # HZ_Bands = [[.001,50],[.01,25],[.1,10],[.1,5],[.5,2]]
# ### Loop through band plots
# for idx in np.arange(0,5,1):
#     merge_dfTEMP = merge_df[(merge_df.MinHZ==HZ_Bands[idx][0]) & (merge_df.MaxHz==HZ_Bands[idx][1])]
#     #### Fix n=2
#     ax[idx,0].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_moment,'o', alpha=.1)
#     median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_moment, ax[idx,0])
#     ax[idx,0].set_xlabel('Mw')
#     ax[idx,0].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Mo (Fc n=2) MPa')
#     ax[idx,0].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#     #### n=?
#     ax[idx,1].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_moment, 'o', alpha=.1)
#     median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_moment, ax[idx,1])
#     ax[idx,1].set_xlabel('Mw')
#     ax[idx,1].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Mo  (Fc n=?) MPa')
#     ax[idx,1].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#
# ax[0,0].set_ylim([5e-3,5e1])
# fig.tight_layout()
# fig.savefig(dataPath+'Figures/'+'StressDropTrends_Ratio_Mo.png')



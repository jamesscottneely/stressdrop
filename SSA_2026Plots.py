
##### Plot to look for magnitude/stress drop trends
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import matplotlib.patheffects as pe
from scipy.stats import linregress
import stress_func as sf


def median_bin(xvals, yvals, ax):
    xvals = xvals.to_numpy()
    yvals = yvals.to_numpy()
    shiftval = .25
    intval = shiftval / 2

    m_pts = np.arange(4, 6.75, shiftval)
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
            ax.text(m, median_bin + median_bin * 0.1, "{:.2f}".format(median_bin), va='bottom', ha='center',
                        color='black', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        except:
            continue

def bestfitLine(minmag,maxmag,x_var,y_var,ax,color):
    ### PLot best fit lin
    x_var_trim =x_var[(x_var>=minmag) & (x_var<=maxmag)]
    y_var_trim = y_var[(x_var>=minmag) & (x_var<=maxmag)]
    print(x_var_trim)
    # m, b, r_value, p_value, std_err = linregress(x_var_trim, np.log10(y_var_trim))
    m, b = np.polyfit(x_var_trim, np.log10(y_var_trim), 1)
    x = np.arange(minmag,maxmag,.1)
    ax.plot(x, 10**(m * x + b), color=color, label="[{:.1f},{:.1f}] slope:{:.2f}".format(minmag,maxmag,m))
    return ax

###### DataPath
dcat = 'Brune_0.50SD_100HZ'
dataPath = "/Users/jamesneely/Documents/NSF/StressDrop_Bands/"+dcat+"/"
### Read Data
truth = pd.read_csv(dataPath + "eqFile.txt",sep='|')
estimate = pd.read_csv(dataPath + "FittingFile.csv")

### Merge files
merge_df = truth.merge(estimate, how='inner',on='eq_id')
### Create ratios



####################
#### Create plot type 1
####################
# fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,6),sharey=True)
# ### Plot the truth
# ax[0].semilogy(merge_df[merge_df.MaxHz==50].mw_tot,merge_df[merge_df.MaxHz==50].sigma_tot_area,'o',alpha=.1)
# median_bin(merge_df.mw_tot,merge_df.sigma_tot_area, ax[0])
# # bestfitLine(4,5.5,merge_df.mw_tot,merge_df.sigma_tot_area,ax[0],'red')
# # bestfitLine(4,6,merge_df.mw_tot,merge_df.sigma_tot_area,ax[0],'purple')
# bestfitLine(4,7,merge_df.mw_tot,merge_df.sigma_tot_area,ax[0],'orange')
# med = np.median(merge_df.sigma_tot_area)
# sd_ln = np.std(np.log(merge_df.sigma_tot_area))
# textstr = "Med: {:.1f} ln_sd: {:.1f}".format(med,sd_ln)
# ax[0].text(.99,.01,textstr,ha='right',va='bottom',transform=ax[0].transAxes)
# ax[0].legend()
# ax[0].set_xlabel('Mw')
# ax[0].set_ylabel(r'$\Delta\sigma$ (True - Area) MPa')
# ax[0].set_title("Synth. (Area-Weighted)")
# ylim = [5e-2,3e1]
# ax[0].set_ylim(ylim)
# ### Plot the truth
# ax[1].semilogy(merge_df[merge_df.MaxHz==50].mw_tot,merge_df[merge_df.MaxHz==50].sigma_tot_moment,'o',alpha=.1)
# median_bin(merge_df.mw_tot,merge_df.sigma_tot_moment, ax[1])
# # bestfitLine(4,5.5,merge_df.mw_tot,merge_df.sigma_tot_moment,ax[1],'red')
# # bestfitLine(4,6,merge_df.mw_tot,merge_df.sigma_tot_moment,ax[1],'purple')
# bestfitLine(4,7,merge_df.mw_tot,merge_df.sigma_tot_moment,ax[1],'orange')
# med = np.median(merge_df.sigma_tot_moment)
# sd_ln = np.std(np.log(merge_df.sigma_tot_moment))
# textstr = "Med: {:.1f} ln_sd: {:.1f}".format(med,sd_ln)
# ax[1].text(.99,.01,textstr,ha='right',va='bottom',transform=ax[1].transAxes)
# ax[1].set_xlabel('Mw')
# ax[1].set_ylabel(r'$\Delta\sigma$ (True - Mo) MPa')
# ax[1].set_title("Synth. (Mo-Weighted)")
# # ax[1].set_ylim(ylim)
# ax[1].legend()
# fig.tight_layout()
#
#
# fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/'+'StressDropTrends_GR.png')
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######

####### ####### ####### ####### ####### #######
####### PLot 2: What are are measuring
####### ####### ####### ####### ####### #######
# fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6),sharey=True)
# ### Plot the truth
# print(merge_df.iloc[0])
# ax[0].loglog(merge_df[merge_df.MaxHz==50].SIG_2_Mo,merge_df[merge_df.MaxHz==50].sigma_tot_area,'o',alpha=.1)
# med = np.median(merge_df[merge_df.MaxHz==50].SIG_2_Mo/merge_df[merge_df.MaxHz==50].sigma_tot_area)
# sd_ln = np.std(merge_df[merge_df.MaxHz==50].SIG_2_Mo/merge_df[merge_df.MaxHz==50].sigma_tot_area)
# corr = np.corrcoef(merge_df[merge_df.MaxHz==50].SIG_2_Mo,merge_df[merge_df.MaxHz==50].sigma_tot_area)[0,1]
# textstr = "Med Ratio: {:.1f} SD Ratio: {:.1f} Corr: {:.2f}".format(med,sd_ln,corr)
# ax[0].text(.99,.01,textstr,ha='right',va='bottom',transform=ax[0].transAxes)
# ax[0].set_xlabel(r'$\Delta\sigma$ MPa')
# ax[0].set_ylabel(r'$\Delta\sigma$ (True - Area) MPa')
# axx = ax[0].get_xlim()
# axy = ax[0].get_ylim()
# axmin = min(axx[0],axy[0])
# axmax = min(axx[1],axy[1])
# ax[0].plot([axmin,axmax],[axmin,axmax],ls='--',lw=1)
#
# ### Plot the truth
# ax[1].loglog(merge_df[merge_df.MaxHz==50].SIG_2_Mo,merge_df[merge_df.MaxHz==50].sigma_tot_moment,'o',alpha=.1)
# med = np.median(merge_df[merge_df.MaxHz==50].SIG_2_Mo/merge_df[merge_df.MaxHz==50].sigma_tot_moment)
# sd_ln = np.std(merge_df[merge_df.MaxHz==50].SIG_2_Mo/merge_df[merge_df.MaxHz==50].sigma_tot_moment)
# corr = np.corrcoef(merge_df[merge_df.MaxHz==50].SIG_2_Mo,merge_df[merge_df.MaxHz==50].sigma_tot_moment)[0,1]
# textstr = "Med Ratio: {:.1f} SD Ratio: {:.1f} Corr: {:.2f}".format(med,sd_ln,corr)
# ax[1].text(.99,.01,textstr,ha='right',va='bottom',transform=ax[1].transAxes)
# ax[1].set_xlabel(r'$\Delta\sigma$ MPa')
# ax[1].set_ylabel(r'$\Delta\sigma$ (True - Moment) MPa')
# ax[1].plot([axmin,axmax],[axmin,axmax],ls='--',lw=1)
# fig.tight_layout()
# fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/'+'StressDrop_Measure.png')
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######

####### ####### ####### ####### ####### #######
####### PLot 3: Impact of Limited bandwidth
####### ####### ####### ####### ####### #######
# HZ_Bands= [[.001,50],[.001,10],[.001,2]]
# fig,ax = plt.subplots(nrows=3,ncols=2,figsize=(14,15))
# ### Loop through band plots
# for idx in np.arange(0,3,1):
#     print(idx)
#     ax[idx,0].set_ylim([3e-2,4e1])
#     merge_dfTEMP = merge_df[(merge_df.MinHZ==HZ_Bands[idx][0]) & (merge_df.MaxHz==HZ_Bands[idx][1])]
#     #### Fix n=2
#     ax[idx,0].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo,'o', alpha=.1)
#     median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo, ax[idx,0])
#     bestfitLine(4, 5.5, merge_dfTEMP.mw_tot,  merge_dfTEMP.SIG_2_Mo, ax[idx, 0], 'red')
#     bestfitLine(4, 6, merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo, ax[idx, 0], 'purple')
#     bestfitLine(4, 7, merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo, ax[idx, 0], 'orange')
#
#     ax[idx, 0].legend()
#     ax[idx,0].set_xlabel('Mw')
#     ax[idx,0].set_ylabel(r'$\Delta\sigma$ MPa')
#     ax[idx,0].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#
#     #### PLot FC values
#     ax[idx,1].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.Fc_2_Mo, 'o', alpha=.1)
#     median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.Fc_2_Mo, ax[idx,1])
#     ax[idx,1].set_xlabel('Mw')
#     ax[idx,1].set_ylabel(r'$f_c$ (Hz)')
#     # ax[idx+1,1].axhline(HZ_Bands[idx][0],ls='--',color='red')
#     ax[idx,1].axhline(HZ_Bands[idx][1],ls='--',color='red')
#     ax[idx,1].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#     ylim_fc = [1e-2,2e1]
#     ax[idx, 1].set_ylim(ylim_fc)
#
# ax[0,0].set_ylim([5e-3,5e1])
# fig.tight_layout()
# fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/'+'SSA_FITBound_Measure.png')
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######

####### ####### ####### ####### ####### #######
####### PLot 5: Impact of Free moment fit
####### ####### ####### ####### ####### #######
# HZ_Bands= [[.001,50],[.001,10],[.001,2]]
# fig,ax = plt.subplots(nrows=3,ncols=2,figsize=(14,15))
# ### Loop through band plots
# for idx in np.arange(0,3,1):
#     print(idx)
#     ax[idx,0].set_ylim([3e-2,4e1])
#     merge_dfTEMP = merge_df[(merge_df.MinHZ==HZ_Bands[idx][0]) & (merge_df.MaxHz==HZ_Bands[idx][1])]
#     #### Fix n=2
#     ax[idx,0].semilogy(sf.mo2mw(merge_dfTEMP.Est_MoFree), merge_dfTEMP.SIG_2_MoFree,'o', alpha=.1)
#     median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_MoFree, ax[idx,0])
#     bestfitLine(4, 5.5, sf.mo2mw(merge_dfTEMP.Est_MoFree),  merge_dfTEMP.SIG_2_MoFree, ax[idx, 0], 'red')
#     bestfitLine(4, 6, sf.mo2mw(merge_dfTEMP.Est_MoFree), merge_dfTEMP.SIG_2_MoFree, ax[idx, 0], 'purple')
#     bestfitLine(4, 7, sf.mo2mw(merge_dfTEMP.Est_MoFree), merge_dfTEMP.SIG_2_MoFree, ax[idx, 0], 'orange')
#
#     ax[idx, 0].legend()
#     ax[idx,0].set_xlabel('Mw')
#     ax[idx,0].set_ylabel(r'$\Delta\sigma$ MPa')
#     ax[idx,0].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#
#     #### PLot FC values
#     ax[idx,1].semilogy(sf.mo2mw(merge_dfTEMP.Est_MoFree), merge_dfTEMP.Fc_2_MoFree, 'o', alpha=.1)
#     median_bin(sf.mo2mw(merge_dfTEMP.Est_MoFree), merge_dfTEMP.Fc_2_MoFree, ax[idx,1])
#     ax[idx,1].set_xlabel('Mw')
#     ax[idx,1].set_ylabel(r'$f_c$ (Hz)')
#     # ax[idx+1,1].axhline(HZ_Bands[idx][0],ls='--',color='red')
#     ax[idx,1].axhline(HZ_Bands[idx][1],ls='--',color='red')
#     ax[idx,1].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#     ylim_fc = [1e-2,2e1]
#     ax[idx, 1].set_ylim(ylim_fc)
#
# ax[0,0].set_ylim([5e-3,5e1])
# fig.tight_layout()
# fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/'+'SSA_FITBound_Measure_MoFree.png')
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######


####### ####### ####### ####### ####### #######
####### Plot 6: EGF (1 MPa) Comparison - Fixed Mo
####### ####### ####### ####### ####### #######

##### Pull EGFs
# EGFs = pd.read_csv(dataPath +"/EGF_1MPa/EGF_1MPa.txt")
# finalMerge = pd.merge(truth,EGFs,right_on=['MAIN_ID'],left_on=['eq_id'],how='left',suffixes=('_Truth','_RatioFit'))
# print(finalMerge)
# fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,16),sharex=True)
# ### Plot the truth
# HZ_Bands= [[.001,50],[.001,10],[.001,2]]
# ### Loop through band plots
# for idx in np.arange(0,3,1):
#     print(idx)
#     merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
#     #### Fix n=2
#     ax[idx].loglog(merge_dfTEMP.SIG_2_Mo_Main_comp_simp, merge_dfTEMP.SIG_2_Mo_EGF_simp_comp,'o', alpha=.1)
#     ax[idx].set_ylabel(r'$\Delta\sigma$ (As EGF)')
#     ax[idx].set_xlabel(r'$\Delta\sigma$ (As Main)')
#     ax[idx].set_title("Spec Ratio Comp=Main vs EGF Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#     ax[idx].plot([5e-3, 5e1], [5e-3, 5e1], ls='--', color='red')
# ax[0].set_ylim([5e-3,5e1])
# ax[0].set_xlim([5e-3,5e1])
# fig.tight_layout()
# fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/SSA_EGF_Simp_MoFixed_'+dcat+'.png')
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######



####### ####### ####### ####### ####### #######
####### Plot 7: EGF (1 MPa) Comparison - Free Mo
####### ####### ####### ####### ####### #######

##### Pull EGFs
# EGFs = pd.read_csv(dataPath +"/EGF_1MPa/EGF_1MPa.txt")
# finalMerge = pd.merge(truth,EGFs,right_on=['MAIN_ID'],left_on=['eq_id'],how='left',suffixes=('_Truth','_RatioFit'))
# print(finalMerge)
# fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,16),sharex=True)
# ### Plot the truth
# HZ_Bands= [[.001,50],[.001,10],[.001,2]]
# ### Loop through band plots
# for idx in np.arange(0,3,1):
#     print(idx)
#     merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
#     #### Fix n=2
#     ax[idx].loglog(merge_dfTEMP.SIG_2_MoFree_Main_comp_simp, merge_dfTEMP.SIG_2_MoFree_EGF_simp_comp,'o', alpha=.1)
#     ax[idx].set_ylabel(r'$\Delta\sigma$ (As EGF)')
#     ax[idx].set_xlabel(r'$\Delta\sigma$ (As Main)')
#     ax[idx].set_title("Spec Ratio Comp=Main vs EGF Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#     ax[idx].plot([5e-3, 5e1], [5e-3, 5e1], ls='--', color='red')
# ax[0].set_ylim([5e-3,5e1])
# ax[0].set_xlim([5e-3,5e1])
# fig.tight_layout()
# fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/SSA_EGF_Simp_MoFree_'+dcat+'.png')
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######


####### ####### ####### ####### ####### #######
####### Plot 8: EGF (complex) Comparison - fixed Mo
####### ####### ####### ####### ####### #######

##### Pull EGFs
# EGFs = pd.read_csv(dataPath +"EGFs/EGFs.txt")
# finalMerge = pd.merge(truth,EGFs,right_on=['MAIN_ID'],left_on=['eq_id'],how='left',suffixes=('_Truth','_RatioFit'))
# print(finalMerge)
# fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,16),sharex=True)
# ### Plot the truth
# HZ_Bands= [[.001,50],[.001,10],[.001,2]]
# ### Loop through band plots
# for idx in np.arange(0,3,1):
#     print(idx)
#     merge_dfTEMP = finalMerge[(finalMerge.MinHZ==HZ_Bands[idx][0]) & (finalMerge.MaxHz==HZ_Bands[idx][1])]
#     grouped_df_MAIN = merge_dfTEMP.groupby(['MAIN_ID'])['SIG_2_Mo_Main'].apply(scipy.stats.gmean).reset_index()
#     grouped_df_EGF = merge_dfTEMP.groupby(['EGF_ID'])['SIG_2_Mo_EGF'].apply(scipy.stats.gmean).reset_index()
#     regroup_final = grouped_df_MAIN.merge(grouped_df_EGF,left_on='MAIN_ID',right_on='EGF_ID',how='inner')
#     #### Fix n=2
#     ax[idx].loglog(regroup_final.SIG_2_Mo_Main, regroup_final.SIG_2_Mo_EGF,'o', alpha=.1)
#     ax[idx].set_ylabel(r'$\Delta\sigma$ (As EGF)')
#     ax[idx].set_xlabel(r'$\Delta\sigma$ (As Main)')
#     ax[idx].set_title("Spec Ratio Comp=Main vs EGF Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
#     ax[idx].plot([5e-3, 5e1], [5e-3, 5e1], ls='--', color='red')
# ax[0].set_ylim([5e-3,5e1])
# ax[0].set_xlim([5e-3,5e1])
# fig.tight_layout()
# fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/SSA_EGF_Complex_MoFixed_'+dcat+'.png')


####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######


####### ####### ####### ####### ####### #######
####### Plot 8: EGF (complex) Comparison - fixed Mo
####### ####### ####### ####### ####### #######

##### Pull EGFs
EGFs = pd.read_csv(dataPath +"EGFs/EGFs.txt")
# finalMerge = pd.merge(truth,EGFs,right_on=['MAIN_ID'],left_on=['eq_id'],how='left',suffixes=('_Truth','_RatioFit'))
print(EGFs)
print(truth[truth.eq_id=='202508221755893488_3918'].iloc[])
fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,16),sharex=True)
### Plot the truth
HZ_Bands= [[.001,50],[.001,10],[.001,2]]
### Loop through band plots
for idx in np.arange(0,3,1):
    print(idx)
    merge_dfTEMP = EGFs[(EGFs.MinHZ==HZ_Bands[idx][0]) & (EGFs.MaxHz==HZ_Bands[idx][1])]
    grouped_df_MAIN = merge_dfTEMP.groupby(['MAIN_ID'])['SIG_2_MoFree_Main'].apply(scipy.stats.gmean).reset_index()
    grouped_df_EGF = merge_dfTEMP.groupby(['EGF_ID'])['SIG_2_MoFree_EGF'].apply(scipy.stats.gmean).reset_index()
    regroup_final = grouped_df_MAIN.merge(grouped_df_EGF,left_on='MAIN_ID',right_on='EGF_ID',how='inner')
    print(regroup_final)
    #### Fix n=2
    ax[idx].loglog(regroup_final.SIG_2_MoFree_Main, regroup_final.SIG_2_MoFree_EGF,'o', alpha=.1)
    ax[idx].set_ylabel(r'$\Delta\sigma$ (As EGF)')
    ax[idx].set_xlabel(r'$\Delta\sigma$ (As Main)')
    ax[idx].set_title("Spec Ratio Comp=Main vs EGF Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
    ax[idx].plot([5e-3, 5e1], [5e-3, 5e1], ls='--', color='red')
ax[0].set_ylim([5e-3,5e1])
ax[0].set_xlim([5e-3,5e1])
fig.tight_layout()
fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/SSA_2026_Figures/SSA_EGF_Complex_MoFree_'+dcat+'.png')


####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######
####### ####### ####### ####### ####### #######

##### Plot to look for magnitude/stress drop trends
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import matplotlib.patheffects as pe
from scipy.stats import linregress


def median_bin(xvals, yvals, ax):
    xvals = xvals.to_numpy()
    yvals = yvals.to_numpy()
    shiftval = .25
    intval = shiftval / 2

    m_pts = np.arange(4, 8.25, shiftval)
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
dataPath = "/Users/jamesneely/Documents/NSF/StressDrop_Bands/Brune_0.50SD_100HZ/"
### Read Data
truth = pd.read_csv(dataPath + "eqFile.txt",sep='|')
estimate = pd.read_csv(dataPath + "FittingFile.csv")

### Merge files
merge_df = truth.merge(estimate, how='inner',on='eq_id')
### Create ratios

print(merge_df.iloc[0])
####################
#### Create plot type 1
####################
fig,ax = plt.subplots(nrows=6,ncols=2,figsize=(15,25))
### Plot the truth
ax[0,0].semilogy(merge_df[merge_df.MaxHz==50].mw_tot,merge_df[merge_df.MaxHz==50].sigma_tot_area,'o',alpha=.1)
median_bin(merge_df.mw_tot,merge_df.sigma_tot_area, ax[0,0])
bestfitLine(4,5.5,merge_df.mw_tot,merge_df.sigma_tot_area,ax[0,0],'red')
bestfitLine(4,6,merge_df.mw_tot,merge_df.sigma_tot_area,ax[0,0],'purple')
bestfitLine(4,7,merge_df.mw_tot,merge_df.sigma_tot_area,ax[0,0],'orange')

ax[0,0].legend()
ax[0,0].set_xlabel('Mw')
ax[0,0].set_ylabel(r'$\Delta\sigma$ (True - Area) MPa')
ax[0,0].set_title("Truth (Area-Weighted)")
ylim = [1e-2,4e1]
ax[0,0].set_ylim(ylim)
### Plot the truth
ax[0,1].semilogy(merge_df[merge_df.MaxHz==50].mw_tot,merge_df[merge_df.MaxHz==50].sigma_tot_moment,'o',alpha=.1)
median_bin(merge_df.mw_tot,merge_df.sigma_tot_moment, ax[0,1])
bestfitLine(4,5.5,merge_df.mw_tot,merge_df.sigma_tot_moment,ax[0,1],'red')
bestfitLine(4,6,merge_df.mw_tot,merge_df.sigma_tot_moment,ax[0,1],'purple')
bestfitLine(4,7,merge_df.mw_tot,merge_df.sigma_tot_moment,ax[0,1],'orange')

ax[0,1].set_xlabel('Mw')
ax[0,1].set_ylabel(r'$\Delta\sigma$ (True - Mo) MPa')
ax[0,1].set_title("Truth (Mo-Weighted)")
ax[0,1].set_ylim(ylim)
ax[0,1].legend()
HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
# HZ_Bands = [[.001,50],[.01,25],[.1,10],[.1,5],[.5,2]]
### Loop through band plots
for idx in np.arange(0,5,1):
    print(idx)
    merge_dfTEMP = merge_df[(merge_df.MinHZ==HZ_Bands[idx][0]) & (merge_df.MaxHz==HZ_Bands[idx][1])]
    #### Fix n=2
    ax[idx+1,0].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo,'o', alpha=.1)
    median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo, ax[idx+1,0])
    bestfitLine(4, 5.5, merge_dfTEMP.mw_tot,  merge_dfTEMP.SIG_2_Mo, ax[idx+1, 0], 'red')
    bestfitLine(4, 6, merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo, ax[idx+1, 0], 'purple')
    bestfitLine(4, 7, merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo, ax[idx+1, 0], 'orange')

    ax[idx+1, 0].legend()
    ax[idx+1,0].set_xlabel('Mw')
    ax[idx+1,0].set_ylabel(r'$\Delta\sigma$ (Fc n=2)')
    ax[idx+1,0].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
    ax[idx+1,0].set_ylim(ylim)
    #### PLot FC values
    ax[idx+1,1].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.Fc_2_Mo, 'o', alpha=.1)
    median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.Fc_2_Mo, ax[idx+1,1])
    ax[idx+1,1].set_xlabel('Mw')
    ax[idx+1,1].set_ylabel(r'$f_c$ (Hz)')
    # ax[idx+1,1].axhline(HZ_Bands[idx][0],ls='--',color='red')
    ax[idx+1,1].axhline(HZ_Bands[idx][1],ls='--',color='red')
    ax[idx+1,1].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
    ylim_fc = [3e-3,2e1]
    ax[idx + 1, 1].set_ylim(ylim_fc)

ax[0,0].set_ylim([5e-3,5e1])
fig.tight_layout()
fig.savefig(dataPath+'Figures/'+'StressDropTrends.png')

####################
#### Create plot type 2
####################
fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(15,25),sharey=True)
### Plot the truth
# HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
# HZ_Bands = [[.001,50],[.01,25],[.1,10],[.1,5],[.5,2]]

### Loop through band plots
for idx in np.arange(0,5,1):
    merge_dfTEMP = merge_df[(merge_df.MinHZ==HZ_Bands[idx][0]) & (merge_df.MaxHz==HZ_Bands[idx][1])]
    #### Fix n=2
    ax[idx,0].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_area,'o', alpha=.1)
    median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_area, ax[idx,0])
    ax[idx,0].set_xlabel('Mw')
    ax[idx,0].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Area (Fc n=2)')
    ax[idx,0].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
    #### n=?
    ax[idx,1].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_area, 'o', alpha=.1)
    median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_area, ax[idx,1])
    ax[idx,1].set_xlabel('Mw')
    ax[idx,1].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Area  (Fc n=?)')
    ax[idx,1].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))

ax[0,0].set_ylim([1e-1,5e1])
fig.tight_layout()
fig.savefig(dataPath+'Figures/'+'StressDropTrends_Ratio_Area.png')

####################
#### Create plot type 3
####################
fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(15,25),sharey=True)
### Plot the truth
# HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
# HZ_Bands = [[.001,50],[.01,25],[.1,10],[.1,5],[.5,2]]
### Loop through band plots
for idx in np.arange(0,5,1):
    merge_dfTEMP = merge_df[(merge_df.MinHZ==HZ_Bands[idx][0]) & (merge_df.MaxHz==HZ_Bands[idx][1])]
    #### Fix n=2
    ax[idx,0].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_moment,'o', alpha=.1)
    median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_2_Mo/merge_dfTEMP.sigma_tot_moment, ax[idx,0])
    ax[idx,0].set_xlabel('Mw')
    ax[idx,0].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Mo (Fc n=2)')
    ax[idx,0].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))
    #### n=?
    ax[idx,1].semilogy(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_moment, 'o', alpha=.1)
    median_bin(merge_dfTEMP.mw_tot, merge_dfTEMP.SIG_n_Mo/merge_dfTEMP.sigma_tot_moment, ax[idx,1])
    ax[idx,1].set_xlabel('Mw')
    ax[idx,1].set_ylabel(r'$\Delta\sigma$/$\Delta\sigma$Mo  (Fc n=?)')
    ax[idx,1].set_title("Frequency Range [{:.1e},{:.1e}] Hz".format(HZ_Bands[idx][0],HZ_Bands[idx][1]))

ax[0,0].set_ylim([1e-1,5e1])
fig.tight_layout()
fig.savefig(dataPath+'Figures/'+'StressDropTrends_Ratio_Mo.png')



#!/usr/bin/env python3  

#### This script plots stress drop histograms
import numpy as np
import pandas as pd
import stress_func as sf
import scipy.optimize as optimize

### File
eqdirs = 'Brune_0.50SD_100HZ_Uniform'
mag_dif = 1.5# Minimum magnitude difference for earthquakes

### Read data
path = '/Users/jamesneely/Documents/NSF/StressDrop_Bands/' + eqdirs
eqfile = path+'/eqFile.txt'
eqdata = pd.read_csv(eqfile,sep='|')
eqdata.sort_values(by='mw_tot',ascending=False,inplace=True) # Sort in ascending order

###### Stress Drop Variables
beta = 3600. # s-wave velocity m/s
alpha = beta*np.sqrt(3) # p-wave velocity m/s
rup_vel = .9*beta # rupture velocity
ro = 2800 # density kg/m^3
mu = .5*1e9 # shear modulus
k = 0.37 # Shape factor brune calculation
c = 1 # Constant for Fc to T conversion
cval = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5)))) # constant parameter

##### Frequency Specific Variables
log_samp = 0.025 # Log sampling for fitting spectral model
HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]

##### Create Out Pandas tables for data
dataout_list = []
count_id = 0
tot_num = len(eqdata)
plot_count = 0
for index, row in eqdata.iterrows():
    egf_table = {}
    #### Find EGFs for smaller earthquakes
    egf_IDs = eqdata[row['mw_tot']-eqdata['mw_tot']>=mag_dif].eq_id
    if len(egf_IDs) < 50:
        n_samp = len(egf_IDs)
    else:
        n_samp = 50
    egf_IDs = egf_IDs.sample(n=n_samp)
    print(" EQ {:.0f}/{:.0f}".format(count_id+1,tot_num))
    if len(egf_IDs) == 0:
        break
    else:
        #### Read Main STF File
        STF_main = pd.read_csv(path+'/STFs/' +row['eq_id'] + ".txt")
        samp_rate = STF_main['time'][1]
        STF_main_FFT = np.fft.rfft(STF_main['STF'])*samp_rate # Convert to freq domain
        Freq_fft = np.fft.rfftfreq(len(STF_main['time']),d=samp_rate)
        ratio_array = np.zeros((len(egf_IDs),len(Freq_fft))) # Initialize array for stacking
        egf_mom_array = np.zeros(len(egf_IDs)) # Initialize array for EGF moment average
        #### Loop through EGF files
        for egf in egf_IDs:
            STF_egf = pd.read_csv(path+'/STFs/' +egf + ".txt")
            #### Convert to frequency domain
            STF_egf_FFT = np.fft.rfft(STF_egf['STF'])*samp_rate # Convert to freq domain
            #### Spectral ratio
            ratio = np.abs(STF_main_FFT/STF_egf_FFT)
            #### Fit Spectral Ratio
            for Hz in HZ_Bands:
                #### Trim the files to  Hz Band
                low_idx = np.searchsorted(Freq_fft, Hz[0])
                hi_idx = np.searchsorted(Freq_fft, Hz[1])
                Freq_fft_trim = Freq_fft[low_idx:hi_idx]
                ratio_trim = ratio[low_idx:hi_idx]
                #### Resample in log space
                freq_smooth, ratio_smooth = sf.resampSpec(Freq_fft_trim,ratio_trim, log_samp)
                #### Fiting scenario 1: Assume known moment ratio
                ### Assume fixed decays of 2
                initialGuess = [sf.brune_fc(beta, sf.brune_radii(row['moment_tot'], 1), k),
                                sf.brune_fc(beta, sf.brune_radii(eqdata[eqdata['eq_id'] == egf].moment_tot.item(), 1),
                                k)]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
                outvalMoFixed = optimize.least_squares(sf.ratio_bruneModInvMoFixed, initialGuess, bounds=(.00001, np.inf),
                                             args=(ratio_smooth, freq_smooth, row['moment_tot'],
                                                   eqdata[eqdata['eq_id'] == egf].moment_tot.item()))
                est_fc_MoFixed_Main = outvalMoFixed.x[0]
                est_fc_MoFixed_EGF = outvalMoFixed.x[1]
                sigma_fc_est_Mofix_Main = sf.stressCircfc(est_fc_MoFixed_Main, row['moment_tot'], k, beta) # main
                sigma_fc_est_Mofix_EGF = sf.stressCircfc(est_fc_MoFixed_Main, eqdata[eqdata['eq_id'] == egf].moment_tot.item(), k, beta) # egf

                #### #### #### #### #### #### moment ratio unknown and fixed decay of 2
                initialGuess = [sf.brune_fc(beta, sf.brune_radii(row['moment_tot'], 1), k),
                                sf.brune_fc(beta, sf.brune_radii(eqdata[eqdata['eq_id'] == egf].moment_tot.item(), 1),k), 1]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
                outvalMoFree = optimize.least_squares(sf.ratio_bruneModInvMoFree, initialGuess, bounds=(.00001, np.inf),
                                                      args=(ratio_smooth, freq_smooth))
                est_fc_MoFree_Main = outvalMoFree.x[0]
                est_fc_MoFree_EGF = outvalMoFree.x[1]
                MoFree_comp_simp = outvalMoFree.x[2]
                sigma_fc_est_MoFree_Main = sf.stressCircfc(est_fc_MoFree_Main, row['moment_tot'], k,
                                                                     beta)  # main
                sigma_fc_est_MoFree_EGF = sf.stressCircfc(est_fc_MoFree_EGF, eqdata[eqdata['eq_id'] == egf].moment_tot.item(), k,
                                                                    beta)  # egf


                #### Save Data
                dataout_list.append(
                    {'MAIN_ID': row.eq_id, 'EGF_ID':egf,'MinHZ': Hz[0], 'MaxHz': Hz[1], 'Fc_2_Mo_Main': est_fc_MoFixed_Main, 'SIG_2_Mo_Main': sigma_fc_est_Mofix_Main, 'Fc_2_Mo_EGF': est_fc_MoFixed_EGF, 'SIG_2_Mo_EGF': sigma_fc_est_Mofix_EGF,
                     'Fc_2_MoFree_Main': est_fc_MoFree_Main, 'SIG_2_MoFree_Main': sigma_fc_est_MoFree_Main, 'Fc_2_MoFree_EGF': est_fc_MoFree_EGF, 'SIG_2_MoFree_EGF': sigma_fc_est_MoFree_EGF,'EstMo_Ratio':MoFree_comp_simp})


    count_id+=1

##### Save Data files
egf_table_df = pd.DataFrame(dataout_list)
print(egf_table_df)
egffile = path + '/EGFs' + '/' + 'EGFs.txt'
egf_table_df.to_csv(egffile, sep=',', index=False)





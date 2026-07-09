#!/usr/bin/env python3


#This code reads in each earthquake - then using estimates it first as a main, then secondary eq. a 1 mpa stress drop as alternates()

##### For each Synthetic rearth
import numpy as np
import pandas as pd
import stress_func as sf
import scipy.optimize as optimize

### File
eqdirs = 'Brune_0.50SD_100HZ'
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
ref_stress = 1 # reference stress drop value for synthetic

##### Frequency Specific Variables
log_samp = 0.025 # Log sampling for fitting spectral model
# HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
HZ_Bands= [[.001,50],[.01,10],[.1,2]]
##### Create Out Pandas tables for data
dataout_list = []
tot_num = len(eqdata)
for index, row in eqdata.iterrows():

    #### Read Main STF File
    STF_main = pd.read_csv(path+'/STFs/' +row['eq_id'] + ".txt")
    samp_rate = STF_main['time'][1]
    STF_main_FFT = np.fft.rfft(STF_main['STF'])*samp_rate # Convert to freq domain
    Freq_fft = np.fft.rfftfreq(len(STF_main['time']),d=samp_rate)
    #### Generate Synthetic EGF and Main earthquakes
    # print(row)
    fc_main_mo = sf.mw2mo(row['mw_tot']+mag_dif)
    fc_main = sf.stress2fc(ref_stress, fc_main_mo, k, beta) # get fc of main hypothetical
    synthBIG = sf.bruneMod(Freq_fft,[fc_main_mo,fc_main]) # get main synth spec
    fc_egf_mo = sf.mw2mo(row['mw_tot']-mag_dif)
    fc_egf = sf.stress2fc(ref_stress, fc_egf_mo, k, beta) # get fc of egf hypothetical
    synthEGF = sf.bruneMod(Freq_fft,[fc_egf_mo,fc_egf]) # get egf synth spec
    # print(row['moment_tot'])
    # print(row['moment_tot']-sf.mw2mo(row['mw_tot']-mag_dif))
    # print(fc_main,fc_egf)
    #### Spectral ratio
    ratio_comp_simp = np.abs(STF_main_FFT/synthEGF)
    ratio_simp_comp= np.abs(synthBIG/STF_main_FFT)
    #### Fit Spectral Ratio
    for Hz in HZ_Bands:
        #### Trim the files to  Hz Band
        low_idx = np.searchsorted(Freq_fft, Hz[0])
        hi_idx = np.searchsorted(Freq_fft, Hz[1])
        Freq_fft_trim = Freq_fft[low_idx:hi_idx]
        ### Trim ratios
        ratio_comp_simp_trim = ratio_comp_simp[low_idx:hi_idx]
        ratio_simp_comp_trim = ratio_simp_comp[low_idx:hi_idx]

        #### Resample in log space
        freq_comp_simp_smooth, ratio_comp_simp_smooth = sf.resampSpec(Freq_fft_trim,ratio_comp_simp_trim, log_samp)
        freq_simp_comp_smooth, ratio_simp_comp_smooth = sf.resampSpec(Freq_fft_trim, ratio_simp_comp_trim, log_samp)
        #### #### #### #### #### #### Assume known moment ratio and fixed decay of 2
        ############### When complex earthquake is main
        print("Complex = Main")
        initialGuess = [sf.brune_fc(beta, sf.brune_radii(row['moment_tot'], 1), k),
                        fc_egf]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
        print(initialGuess)
        outvalMoFixed = optimize.least_squares(sf.ratio_bruneModInvMoFixed, initialGuess,
                                     args=(ratio_comp_simp_smooth, freq_comp_simp_smooth, row['moment_tot'],
                                          fc_egf_mo))
        est_fc_MoFixed_Main_comp_simp = outvalMoFixed.x[0]
        est_fc_MoFixed_EGF_comp_simp = outvalMoFixed.x[1]
        sigma_fc_est_Mofixed_Main_comp_simp = sf.stressCircfc(est_fc_MoFixed_Main_comp_simp, row['moment_tot'], k, beta) # main
        sigma_fc_est_Mofixed_EGF_comp_simp= sf.stressCircfc(est_fc_MoFixed_EGF_comp_simp, fc_egf_mo, k, beta) # egf
        ############### When complex earthquake is the EGF
        initialGuess = [fc_main,sf.brune_fc(beta, sf.brune_radii(row['moment_tot'], 1), k)]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
        print("Simple Main")
        print(initialGuess)
        outvalMoFixed = optimize.least_squares(sf.ratio_bruneModInvMoFixed, initialGuess,
                                     args=(ratio_simp_comp_smooth, freq_simp_comp_smooth,fc_main_mo,
                                           row['moment_tot']))
        est_fc_MoFixed_Main_simp_comp = outvalMoFixed.x[0]
        est_fc_MoFixed_EGF_simp_comp = outvalMoFixed.x[1]
        sigma_fc_est_Mofixed_Main_simp_comp= sf.stressCircfc(est_fc_MoFixed_Main_simp_comp, fc_main_mo, k, beta) # main
        sigma_fc_est_Mofixed_EGF_simp_comp= sf.stressCircfc(est_fc_MoFixed_EGF_simp_comp, row['moment_tot'], k, beta) # egf
        #### #### #### #### #### #### moment ratio unknown and fixed decay of 2
        ############### When complex earthquake is main
        initialGuess = [sf.brune_fc(beta, sf.brune_radii(row['moment_tot'], 1), k),
                        fc_egf,1]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
        outvalMoFree = optimize.least_squares(sf.ratio_bruneModInvMoFree, initialGuess, bounds=(.00001, np.inf),
                                     args=(ratio_comp_simp_smooth, freq_comp_simp_smooth))
        est_fc_MoFree_Main_comp_simp = outvalMoFree.x[0]
        est_fc_MoFree_EGF_comp_simp = outvalMoFree.x[1]
        MoFree_comp_simp = outvalMoFree.x[2]
        sigma_fc_est_MoFree_Main_comp_simp = sf.stressCircfc(est_fc_MoFree_Main_comp_simp, row['moment_tot'], k, beta) # main
        sigma_fc_est_MoFree_EGF_comp_simp= sf.stressCircfc(est_fc_MoFree_EGF_comp_simp, fc_egf_mo, k, beta) # egf
        ############### When complex earthquake is the EGF
        initialGuess = [fc_main,sf.brune_fc(beta, sf.brune_radii(row['moment_tot'], 1), k),1]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
        outvalMoFree = optimize.least_squares(sf.ratio_bruneModInvMoFree, initialGuess, bounds=(.00001, np.inf),
                                     args=(ratio_simp_comp_smooth, freq_simp_comp_smooth))
        est_fc_MoFree_Main_simp_comp = outvalMoFree.x[0]
        est_fc_MoFree_EGF_simp_comp = outvalMoFree.x[1]
        MoFree_simp_comp = outvalMoFree.x[2]
        sigma_fc_est_MoFree_Main_simp_comp= sf.stressCircfc(est_fc_MoFree_Main_simp_comp, fc_main_mo, k, beta) # main
        sigma_fc_est_MoFree_EGF_simp_comp= sf.stressCircfc(est_fc_MoFree_EGF_simp_comp, row['moment_tot'], k, beta) # egf
        #### Save Data
        dataout_list.append(
            {'MAIN_ID': row.eq_id, 'MinHZ': Hz[0], 'MaxHz': Hz[1], 'Fc_2_Mo_Main_comp_simp': est_fc_MoFixed_Main_comp_simp, 'SIG_2_Mo_Main_comp_simp': sigma_fc_est_Mofixed_Main_comp_simp, 'Fc_2_Mo_EGF_comp_simp': est_fc_MoFixed_EGF_comp_simp, 'SIG_2_Mo_EGF_comp_simp': sigma_fc_est_Mofixed_EGF_comp_simp,
             'Fc_2_Mo_Main_simp_comp': est_fc_MoFixed_Main_simp_comp, 'SIG_2_Mo_Main_simp_comp': sigma_fc_est_Mofixed_Main_simp_comp, 'Fc_2_Mo_EGF_simp_comp': est_fc_MoFixed_EGF_simp_comp, 'SIG_2_Mo_EGF_simp_comp': sigma_fc_est_Mofixed_EGF_simp_comp,
             'Fc_2_MoFree_Main_comp_simp': est_fc_MoFree_Main_comp_simp,
             'SIG_2_MoFree_Main_comp_simp': sigma_fc_est_MoFree_Main_comp_simp,
             'Fc_2_MoFree_EGF_comp_simp': est_fc_MoFree_EGF_comp_simp,
             'SIG_2_MoFree_EGF_comp_simp': sigma_fc_est_MoFree_EGF_comp_simp,
             'Fc_2_MoFree_Main_simp_comp': est_fc_MoFree_Main_simp_comp,
             'SIG_2_MoFree_Main_simp_comp': sigma_fc_est_MoFree_Main_simp_comp,
             'Fc_2_MoFree_EGF_simp_comp': est_fc_MoFree_EGF_simp_comp,
             'SIG_2_MoFree_EGF_simp_comp': sigma_fc_est_MoFree_EGF_simp_comp,
             'MoFree_comp_simp':MoFree_comp_simp,'MoFree_simp_comp':MoFree_simp_comp})

##### Save Data files
egf_table_df = pd.DataFrame(dataout_list)
egffile = path + '/EGF_1MPa' + '/' + 'EGF_1MPa_Bands.txt'
egf_table_df.to_csv(egffile, sep=',', index=False)
print(egf_table_df.iloc[0])





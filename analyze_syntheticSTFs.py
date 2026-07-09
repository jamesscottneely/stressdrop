#!/usr/bin/env python3  

#### This script plots stress drop histograms
import numpy as np
import scipy.optimize as optimize
import scipy.signal as sig
import stress_func as sf
import pandas as pd
from datetime import datetime
import os
import itertools
import scipy.integrate as integrate
import scipy.stats as stats
import shutil


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
# HZ_Bands= [[.001,50],[.001,25],[.001,10],[.001,5],[.001,2]]
HZ_Bands= [[.001,50],[.01,10],[.1,2]] #  fitting range for brune model




####### Path to STFs
DataPath = '/Users/jamesneely/Documents/NSF/StressDrop_Bands/Brune_0.50SD_100HZ_Uniform'

eqfile = DataPath+'/eqFile.txt'
eqdata = pd.read_csv(eqfile,sep='|')

dataout_list = [] # initialize datalist

for index, row in eqdata.iterrows():
    print(row)
    ##### Read in STF
    STF_main = pd.read_csv(DataPath+'/STFs/' +row['eq_id'] + ".txt")
    samp_rate = STF_main['time'][1]
    ######### Frequency Domain
    for Hz in HZ_Bands:##### Loop through values fitting
        #### Perform FT
        freq,amplitude,FT_complex = sf.sig_process(STF_main['time'].to_numpy(),STF_main['STF'].to_numpy(),Hz) # Process the signal to spectra
        #### Smooth spectra
        freq_smooth,amp_smooth = sf.resampSpec(freq,amplitude,log_samp) ## Resample spectra
        ### Fit Fc only
        initialGuess=[1]## Provide initial guess
        outvalMoFixed = optimize.least_squares(sf.bruneModInvMoFixed,initialGuess,bounds=(.0001,np.inf),args=(amp_smooth,freq_smooth,row.moment_tot))
        est_fc_MoFixed = outvalMoFixed.x[0]
        sigma_fc_est_Mofix = sf.stressCircfc(est_fc_MoFixed,row.moment_tot,k,beta)
        ### Fit fc and slope
        initialGuess=[1,2]## Provide initial guess start
        outval = optimize.least_squares(sf.bruneModInvMoFixedslopeFree,initialGuess,bounds=((.0001,.01),(np.inf,np.inf)),args=(amp_smooth,freq_smooth,row.moment_tot)) ## making the actual fit
        est_fc_freeslope = outval.x[0]
        est_fs_freeslope =  outval.x[1]
        sigma_fc_est_freeslope = sf.stressCircfc(est_fc_freeslope,row.moment_tot,k,beta)
        ### Fit fc and Mo
        initialGuess=[1,row.moment_tot*1.05]## Provide initial guess start
        outval = optimize.least_squares(sf.bruneModInvMoFree,initialGuess,bounds=((.0001,.01),(np.inf,np.inf)),args=(amp_smooth,freq_smooth)) ## making the actual fit
        est_fc_MoFree = outval.x[0]
        est_MoFree =  outval.x[1]
        sigma_fc_est_Mofree = sf.stressCircfc(est_fc_MoFree,est_MoFree,k,beta)
        #### Save Data
        dataout_list.append({'eq_id':row.eq_id,'MinHZ':Hz[0],'MaxHz':Hz[1],'Fc_2_Mo':est_fc_MoFixed,'SIG_2_Mo':sigma_fc_est_Mofix,'Fc_n_Mo':est_fc_freeslope,'n_Mo':est_fs_freeslope,'SIG_n_Mo':sigma_fc_est_freeslope,'Fc_2_MoFree':est_fc_MoFree,'SIG_2_MoFree':sigma_fc_est_Mofree,'Est_MoFree':est_MoFree})


dataout = pd.DataFrame(dataout_list)
datafile =  DataPath + '/FittingFile_BANDS.csv'
dataout.to_csv(datafile,sep=',',index=False)








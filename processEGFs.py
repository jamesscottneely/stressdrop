#!/usr/bin/env python3  

#### This script plots stress drop histograms
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import ticker, cm
import matplotlib.colors as colors
import stress_func as sf
# import statsmodels.api as sm
import scipy

### File
eqdirs = 'FOLDER NAME WHERE SYNTHETIC EGFS STORED
mag_dif = 1.5 # Minimum magnitude difference for earthquakes

### Read data
path = 'ENTER DIRECTORY PATH to SAVE DATA' + eqdirs
plotpath = 'ENTER DIRECTORY PATH to SAVE Plots'+ eqdirs
eqfile = path+'/eqFile.txt'
eqdata = pd.read_csv(eqfile,sep='|')
eqdata.sort_values(by='mw_tot',ascending=False,inplace=True) # Sort in ascending order
idx = 0



##### Create Out Pandas tables for data
egf_table = pd.DataFrame(columns=['main_id','egf_id'])


count_id = 0
tot_num = len(eqdata)
plot_count = 0
for index, row in eqdata.iterrows():
	#### Find EGFs for smaller earthquakes
	egf_IDs = eqdata[row['mw_tot']-eqdata['mw_tot']>=mag_dif].eq_id
	print(" EQ {:.0f}/{:.0f}".format(count_id+1,tot_num))
	if len(egf_IDs) == 0:
		break
	else:
		#### Read Main STF File
		STF_main = pd.read_csv(path+'/STFs/' +row['eq_id'] + ".txt")
		samp_rate = STF_main['time'][1]
		STF_main_FFT = np.fft.rfft(STF_main['STF'])*samp_rate # Convert to freq domain
		Freq_fft = np.fft.rfftfreq(len(STF_main['time']),d=samp_rate)
		ratio_array = np.zeros((len(Freq_fft),len(egf_IDs))) # Initialize array for stacking
		egf_mom_array = np.zeros(len(egf_IDs)) # Initialize array for EGF moment average
		#### Loop through EGF files
		egf_idx = 0
		for egf in egf_IDs:
			print(egf_idx,'/',len(egf_IDs))
			STF_egf = pd.read_csv(path+'/STFs/' +egf + ".txt")
			#### Convert to frequency domain
			STF_egf_FFT = np.fft.rfft(STF_egf['STF'])*samp_rate # Convert to freq domain
			#### Spectral ratio
			ratio = STF_main_FFT/STF_egf_FFT

			################################
			#### Analysis 3: Stack raw time series
			################################	
			ratio_array[:,egf_idx] = np.abs(ratio)/np.abs(ratio[0]) # save normalized (by 0th frequency) ratio values
			egf_mom_array[egf_idx] = eqdata[eqdata['eq_id']==egf].moment_tot.item() # EGF moment



			
			############ ############ ############ ############ ############ ############ ############ ############ ############ ############ 
			egf_idx+=1
		#### Stacking the averages
		mean_EGF_mo = np.mean(egf_mom_array)
		ratio_stack = np.mean(ratio_array,axis=1)*(row['moment_tot']/mean_EGF_mo) # stack and multiple to get relative amplitude differences



			
			
		idx +=1
	count_id+=1
# 	if count_id > 5:
# 		break


##### Save Data files
print(egf_table)
egffile =  path + '/egf_table.txt'
egf_table.to_csv(egffile,sep=',',index=False)
mainfile =  path + '/main_table.txt'
main_table.to_csv(mainfile,sep=',',index=False)





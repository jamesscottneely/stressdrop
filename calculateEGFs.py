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
log_samp = .025
k = 0.37 # Shape factor brune calculation
beta = 3600. # m/s
ro = 2800 # density kg/m^3
alpha = beta*np.sqrt(3) # m/s

percent = 0.05
cfac = 1
### Read data
path = 'ENTER DIRECTORY PATH to SAVE DATA' + eqdirs
plotpath = 'ENTER DIRECTORY PATH to SAVE Plots'+ eqdirs
eqfile = path+'/eqFile.txt'
eqdata = pd.read_csv(eqfile,sep='|')
eqdata.sort_values(by='mw_tot',ascending=False,inplace=True) # Sort in ascending order
idx = 0

egf_tab_loc = ['b','c','d','e']
txt_loc = [.01,.16,.31,.46]
colors = ['red','orange','green','purple']
p_egfs = 4

##### Create Out Pandas tables for data
main_table = pd.DataFrame(columns=['eq_id','est_fc_MoFixed_Main_stack','est_fc_MoFixed_EGF_stack',
					'sigma_fc_est_Mofix_stack','est_fc_decay_Main_stack','est_decay_Main_stack',
					'est_fc_decay_EGF_stack','est_decay_EGF_stack','sigma_fc_est_Mofix_decay_stack',
					'est_fc_MoFixed_Main_Brune','est_fc_MoFixed_EGF_Brune','sigma_fc_est_Mofix_Brune',
					'est_fc_decay_Main_Brune','est_decay_Main_Brune','est_fc_decay_EGF_Brune',
					'est_decay_EGF_Brune','sigma_fc_est_Mofix_decay_Brune','mean_EGF_mo',
					'est_fc_MoFixed_Main_mean','est_fc_decay_Main_mean','est_decay_Main_mean',
					'T_durest_mean','BRE_T_time_mean','stress_fc_MoFixed_mean','stress_fc_decay_mean',
					'stress_T_durest_mean','BRE_rat_stack_2_full','BRE_rat_stack_2_cut','BRE_rat_stack_n_full','BRE_rat_stack_n_cut'])
									
egf_table = pd.DataFrame(columns=['eq_id','main_eq_id','est_fc_MoFixed_Main','est_fc_MoFixed_EGF',
					'sigma_fc_est_Mofix','est_fc_decay_Main','est_decay_Main','est_fc_decay_EGF',
					'est_decay_EGF','sigma_fc_est_Mofix_decay','T_durest','sigma_T_est','BRE_T_time','pf','BRE_rat_ind_2_full','BRE_rat_ind_2_cut','BRE_rat_ind_n_full','BRE_rat_ind_n_cut'])


plot_flag = 'N'
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
		maxHz = 1/(2*samp_rate)
		ratio_array = np.zeros((len(Freq_fft),len(egf_IDs))) # Initialize array for stacking
		egf_mom_array = np.zeros(len(egf_IDs)) # Initialize array for EGF moment average
		if plot_count <= 50:
			############ ############ ############ ############ ############ ############ ############ 
			############ Plot some records
			plot_flag = 'Y' # Set plot flag for this earthquake
			fig, ax = plt.subplot_mosaic([
				['a','a','b','c','f','f'],
				['a','a','d','e','f','f'],
				['k','k','k','l','l','l'],
				['k','k','k','l','l','l']],
				figsize=(24,18),layout='constrained')
			### Plot Original STFs
			ax['a'].plot(STF_main['time'],STF_main['STF'])
			ax['f'].loglog(Freq_fft,np.abs(STF_main_FFT))
			t_start_main_idx = np.argmax(STF_main['STF']>0)
			ax['a'].set_xlim([STF_main['time'].loc[t_start_main_idx]*.9,(STF_main['time'].loc[t_start_main_idx]+row['T_dur'])*1.2])
			main_info_txt = r"$M_w$ {:.1f}".format(row['mw_tot']) + "\n" +"T {:.1f} S".format(row['T_dur'])+"\n" +"BRE {:.1f}".format(row['BRE_T_time']) + "\n" +r"$\Delta\sigma_A$ {:.1f} MPa".format(row['sigma_tot_area'])+"\n" +r"$\Delta\sigma_{{Mo}}$ {:.1f} MPa".format(row['sigma_tot_moment'])+"\n" +r"$\Delta\sigma_{{T}}$ {:.1f} MPa".format(row['sigma_T_est'])
			main_info_txt2 = "Main "+"\n" + r"$\Delta\sigma_{{fc}}$ {:.1f} MPa".format(row['sigma_fc_est_Mofix'])+"\n"  +r"$f_c$ {:.3f} Hz".format(row['est_fc_MoFixed']) +"\n" +r"$n$ {:.1f}".format(row['est_fs_freeslope']) 
			ax['a'].text(.99,.99,main_info_txt,ha='right',va='top',transform=ax['a'].transAxes)
			ax['f'].text(.01,.4,main_info_txt2,ha='left',va='top',transform=ax['f'].transAxes)
			############ ############ ############ ############ ############ ############ ############ 
		
		#### Loop through EGF files
		egf_idx = 0
		for egf in egf_IDs:
			print(egf_idx,'/',len(egf_IDs))
			STF_egf = pd.read_csv(path+'/STFs/' +egf + ".txt")
			#### Convert to frequency domain
			STF_egf_FFT = np.fft.rfft(STF_egf['STF'])*samp_rate # Convert to freq domain
			#### Spectral ratio
			ratio = STF_main_FFT/STF_egf_FFT
			ratio_time = np.fft.irfft(ratio,len(STF_main['time']))*1/samp_rate
			################################
			#### Analysis 1: Individual frequency fitting
			################################
			freq_smooth,ratio_smooth = sf.resampSpec(Freq_fft[1:],np.abs(ratio[1:]),log_samp) ## Resample spectra
			### Assume fixed decays of 2
			initialGuess=[sf.brune_fc(beta,sf.brune_radii(row['moment_tot'],1),k),sf.brune_fc(beta,sf.brune_radii(eqdata[eqdata['eq_id']==egf].moment_tot.item(),1),k)]## Provide initial guess assume Stress Drop = 1 MPa for initial guess		
			outvalMoFixed = scipy.optimize.least_squares(sf.ratio_bruneModInvMoFixed,initialGuess,bounds=(.00001,np.inf),args=(ratio_smooth,freq_smooth,row['moment_tot'],eqdata[eqdata['eq_id']==egf].moment_tot.item()))
			est_fc_MoFixed_Main = outvalMoFixed.x[0]
			est_fc_MoFixed_EGF = outvalMoFixed.x[1]
			sigma_fc_est_Mofix = sf.stressCircfc(est_fc_MoFixed_Main,row['moment_tot'],k,beta)
			## Calculate relative energy ratios
			# full
			BRE_rat_ind_2_full = sf.EGF_energy_ratio(row['moment_tot'],est_fc_MoFixed_Main,2,eqdata[eqdata['eq_id']==egf].moment_tot.item(),est_fc_MoFixed_EGF,2,np.abs(ratio),Freq_fft,ro,alpha,beta)
			# trim to EGF corner
			Freq_fft_cut = Freq_fft[Freq_fft<=est_fc_MoFixed_EGF]
			cutInt = len(Freq_fft_cut) 
			ratio_cut = ratio[:cutInt]
			BRE_rat_ind_2_cut = sf.EGF_energy_ratio(row['moment_tot'],est_fc_MoFixed_Main,2,eqdata[eqdata['eq_id']==egf].moment_tot.item(),est_fc_MoFixed_EGF,2,np.abs(ratio_cut),Freq_fft_cut,ro,alpha,beta)

			### Assume variable decays	
			initialGuess=[sf.brune_fc(beta,sf.brune_radii(row['moment_tot'],1),k),2,sf.brune_fc(beta,sf.brune_radii(eqdata[eqdata['eq_id']==egf].moment_tot.item(),1),k),2]## Provide initial guess assume Stress Drop = 1 MPa and decay = 2		
			outval_decay= scipy.optimize.least_squares(sf.ratio_bruneModInvMoFixedslopeFree,initialGuess,bounds=(.00001,np.inf),args=(ratio_smooth,freq_smooth,row['moment_tot'],eqdata[eqdata['eq_id']==egf].moment_tot.item()))
			est_fc_decay_Main = outval_decay.x[0]
			est_decay_Main = outval_decay.x[1]
			est_fc_decay_EGF = outval_decay.x[2]
			est_decay_EGF = outval_decay.x[3]	
			sigma_fc_est_Mofix_decay = sf.stressCircfc(est_fc_decay_Main,row['moment_tot'],k,beta)
			## Calculate relative energy ratios
			# full
			BRE_rat_ind_n_full = sf.EGF_energy_ratio(row['moment_tot'],est_fc_decay_Main,est_decay_Main,eqdata[eqdata['eq_id']==egf].moment_tot.item(),est_fc_decay_EGF,est_decay_EGF,np.abs(ratio),Freq_fft,ro,alpha,beta)
			# trim to EGF corner
			Freq_fft_cut = Freq_fft[Freq_fft<=est_fc_MoFixed_EGF]
			cutInt = len(Freq_fft_cut) 
			ratio_cut = ratio[:cutInt]
			BRE_rat_ind_n_cut = sf.EGF_energy_ratio(row['moment_tot'],est_fc_decay_Main,est_decay_Main,eqdata[eqdata['eq_id']==egf].moment_tot.item(),est_fc_decay_EGF,est_decay_EGF,np.abs(ratio_cut),Freq_fft_cut,ro,alpha,beta)


			################################
			#### Analysis 2: Butterworth t time
			################################		
# 			pf = est_fc_decay_Main*2# Twice Main corner frequency
# 			sos = scipy.signal.butter(10, pf, 'low', fs=1/samp_rate, output='sos') # Create filter
# 			bw_filter_STF = scipy.signal.sosfilt(sos, ratio_time) # Apply filter to the deconvolved signal in the time domain
# 			T_durest,epoints = sf.rupture_Duration_StartUnk(bw_filter_STF,samp_rate,percent)
# 			sigma_T_est = sf.stressCircSTF(T_durest,row['moment_tot'],k,beta,cfac)	
# 			#### generate Brune in Time with same duration
# 			Brune_T_time = sf.brune_time_moment(np.abs(ratio[0]),cfac/T_durest,STF_main['time'].loc[epoints[0]],STF_main['time'])
# 			Brune_T_time[Brune_T_time<0] = 0 # trim the initial part of Brune
# 			#### Calculate Brune Radiated energy
# 			BRE_T_time = sf.rough_fun(Brune_T_time,bw_filter_STF,samp_rate)
			################################
			#### Analysis 3: Stack raw time series
			################################	
			ratio_array[:,egf_idx] = np.abs(ratio)/np.abs(ratio[0]) # save normalized (by 0th frequency) ratio values
			egf_mom_array[egf_idx] = eqdata[eqdata['eq_id']==egf].moment_tot.item() # EGF moment
			################################ Append values to egf table
			egf_table.loc[len(egf_table.index)] = [egf,row['eq_id'],est_fc_MoFixed_Main,est_fc_MoFixed_EGF,sigma_fc_est_Mofix,est_fc_decay_Main,est_decay_Main,est_fc_decay_EGF,est_decay_EGF,sigma_fc_est_Mofix_decay,0,0,0,0,BRE_rat_ind_2_full,BRE_rat_ind_2_cut,BRE_rat_ind_n_full,BRE_rat_ind_n_cut]
			############ ############ ############ ############ ############ ############ ############ ############ ############ ############ 
			############ ############ Plot EGFs
			if egf_idx < p_egfs:
				EGF_info = eqdata.loc[eqdata.eq_id == egf] 
				EGF_info_txt = r"$M_w$ {:.1f}".format(EGF_info['mw_tot'].item()) + "\n" +"T {:.1f} S".format(EGF_info['T_dur'].item())+"\n" +"BRE {:.1f}".format(EGF_info['BRE_T_time'].item()) + "\n" +r"$\Delta\sigma_A$ {:.1f} MPa".format(EGF_info['sigma_tot_area'].item())+"\n" +r"$\Delta\sigma_{{Mo}}$ {:.1f} MPa".format(EGF_info['sigma_tot_moment'].item())+"\n" +r"$\Delta\sigma_{{T}}$ {:.1f} MPa".format(EGF_info['sigma_T_est'].item())
				EGF_info_txt2 = "EGF "+"\n" + r"$\Delta\sigma_{{fcEGF}}$ {:.1f} MPa".format(EGF_info['sigma_fc_est_Mofix'].item())+"\n"  +r"$f_{{cEGF}}$ {:.3f} Hz".format(EGF_info['est_fc_MoFixed'].item()) +"\n" +r"$n_{{EGF}}$ {:.1f}".format(EGF_info['est_fs_freeslope'].item()) 
# 				filt_text = "EGF "+ "\n" +r"$T_{{est}}$ {:.1f} S".format(T_durest) + "\n" + r"$\Delta\sigma_{{Test}}$ {:.1f}".format(sigma_T_est) + "\n" + "BRE {:.1f}".format(BRE_T_time)	
				### #### Plot Raw EGFs
				ax[egf_tab_loc[egf_idx]].plot(STF_egf['time'],STF_egf['STF'],color=colors[egf_idx])
				ax[egf_tab_loc[egf_idx]].text(.01,.99,'EGF '+egf,ha='left',va='top',transform=ax[egf_tab_loc[egf_idx]].transAxes)
				ax[egf_tab_loc[egf_idx]].text(.99,.99,EGF_info_txt,ha='right',va='top',transform=ax[egf_tab_loc[egf_idx]].transAxes)
				ax['f'].loglog(Freq_fft,np.abs(STF_egf_FFT),label=egf,color=colors[egf_idx])
				ax['f'].text(txt_loc[egf_idx],.01,EGF_info_txt2,ha='left',va='bottom',transform=ax['f'].transAxes,color=colors[egf_idx])
				t_start_idx = np.argmax(STF_egf['STF']>0)
				ax['k'].axvline(est_fc_MoFixed_Main,color=colors[egf_idx],ls='--')
# 				ax[egf_tab_loc[egf_idx]].set_xlim([STF_egf['time'].loc[t_start_idx]*.9,(STF_egf['time'].loc[t_start_idx]+EGF_info['T_dur'].item())*1.25])
				ax[egf_tab_loc[egf_idx]].set_xlim([STF_egf['time'].loc[t_start_idx]*.99,50+STF_egf['time'].loc[t_start_idx]*.99])

# 				ax[egf_tab_loc[egf_idx]].set_xlim([480,525])

				#### #### Plot Deconvolved Signals
				####### Lowpass filtered
# 				ax['i'].plot(STF_egf['time'],bw_filter_STF,color=colors[egf_idx])
# 				## Add Text to spectra panel
# 				ax['i'].text(txt_loc[egf_idx],.8,filt_text,ha='left',va='top',transform=ax['i'].transAxes,color=colors[egf_idx])	
				####### Plot Smoothed 
				est_ratio = sf.ratio_bruneModMoFixed(freq_smooth,[est_fc_MoFixed_Main,est_fc_MoFixed_EGF],row['moment_tot'],EGF_info['moment_tot'].item())
				est_ratio_decay = sf.ratio_bruneModMoFixedslopeFree(freq_smooth,[est_fc_decay_Main,est_decay_Main,est_fc_decay_EGF,est_decay_EGF],row['moment_tot'],EGF_info['moment_tot'].item())
				ratio_txt = "EGF "+ "\n" +r"$f_{{c2}}$ {:.3f} Hz".format(est_fc_MoFixed_Main)+ "\n" +r"$\Delta\sigma_{{fc2}}$ {:.1f} MPa".format(sigma_fc_est_Mofix)#+ "\n" +r"$f_{{cn}}$ {:.3f} Hz".format(est_fc_decay_Main)+ "\n" +r"$\Delta\sigma_{{fcn}}$ {:.1f} MPa".format(sigma_fc_est_Mofix_decay) + "\n"+ r"$n_{{Main}}$: {:.1f}".format(est_decay_Main) + "\n"+ r"$n_{{EGF}}$: {:.1f}".format(est_decay_EGF)
				ax['k'].text(txt_loc[egf_idx],.01,ratio_txt,ha='left',va='bottom',transform=ax['k'].transAxes,color=colors[egf_idx])
				ax['k'].loglog(freq_smooth,ratio_smooth,color=colors[egf_idx],alpha=.2)
				ax['k'].loglog(freq_smooth,est_ratio,color=colors[egf_idx],label='none')
# 				ax['k'].loglog(freq_smooth,est_ratio_decay,color=colors[egf_idx],ls='--',label='none')
			
			############ ############ ############ ############ ############ ############ ############ ############ ############ ############ 
			egf_idx+=1
		#### Stacking the averages
		mean_EGF_mo = np.mean(egf_mom_array)
		ratio_stack = np.mean(ratio_array,axis=1)*(row['moment_tot']/mean_EGF_mo) # stack and multiple to get relative amplitude differences
		freq_smooth,amp_smooth_stack = sf.resampSpec(Freq_fft[1:],np.abs(ratio_stack[1:]),log_samp) ## Resample spectra
		### Assume fixed decays of 2
		initialGuess=[sf.brune_fc(beta,sf.brune_radii(row['moment_tot'],1),k),sf.brune_fc(beta,sf.brune_radii(mean_EGF_mo,1),k)]## Provide initial guess assume Stress Drop = 1 MPa for initial guess	
		outvalMoFixed = scipy.optimize.least_squares(sf.ratio_bruneModInvMoFixed,initialGuess,bounds=(.00001,np.inf),args=(amp_smooth_stack,freq_smooth,row['moment_tot'],mean_EGF_mo))
		est_fc_MoFixed_Main_stack = outvalMoFixed.x[0]
		est_fc_MoFixed_EGF_stack = outvalMoFixed.x[1]
		sigma_fc_est_Mofix_stack = sf.stressCircfc(est_fc_MoFixed_Main_stack,row['moment_tot'],k,beta)
		## Calculate relative energy ratios
		# full
		BRE_rat_stack_2_full = sf.EGF_energy_ratio(row['moment_tot'],est_fc_MoFixed_Main_stack,2,mean_EGF_mo,est_fc_MoFixed_EGF_stack,2,np.abs(ratio_stack),Freq_fft,ro,alpha,beta)
		# trim to EGF corner
		Freq_fft_cut = Freq_fft[Freq_fft<=est_fc_MoFixed_EGF_stack]
		cutInt = len(Freq_fft_cut) 
		ratio_cut = ratio_stack[:cutInt]
		BRE_rat_stack_2_cut = sf.EGF_energy_ratio(row['moment_tot'],est_fc_MoFixed_Main_stack,2,mean_EGF_mo,est_fc_MoFixed_EGF_stack,2,np.abs(ratio_cut),Freq_fft_cut,ro,alpha,beta)
		### Assume variable decays	
		initialGuess=[sf.brune_fc(beta,sf.brune_radii(row['moment_tot'],1),k),2,sf.brune_fc(beta,sf.brune_radii(mean_EGF_mo,1),k),2]## Provide initial guess assume Stress Drop = 1 MPa and decay = 2		
		outval_decay= scipy.optimize.least_squares(sf.ratio_bruneModInvMoFixedslopeFree,initialGuess,bounds=(.00001,np.inf),args=(amp_smooth_stack,freq_smooth,row['moment_tot'],mean_EGF_mo))
		est_fc_decay_Main_stack = outval_decay.x[0]
		est_decay_Main_stack = outval_decay.x[1]
		est_fc_decay_EGF_stack = outval_decay.x[2]
		est_decay_EGF_stack = outval_decay.x[3]	
		sigma_fc_est_Mofix_decay_stack = sf.stressCircfc(est_fc_decay_Main_stack,row['moment_tot'],k,beta)
		## Calculate relative energy ratios
		# full
		BRE_rat_stack_n_full = sf.EGF_energy_ratio(row['moment_tot'],est_fc_decay_Main_stack,est_decay_Main_stack,mean_EGF_mo,est_fc_decay_EGF_stack,est_decay_EGF_stack,np.abs(ratio_stack),Freq_fft,ro,alpha,beta)
		# trim to EGF corner
		Freq_fft_cut = Freq_fft[Freq_fft<=est_fc_decay_EGF_stack]
		cutInt = len(Freq_fft_cut) 
		ratio_cut = ratio_stack[:cutInt]
		BRE_rat_stack_n_cut = sf.EGF_energy_ratio(row['moment_tot'],est_fc_decay_Main_stack,est_decay_Main_stack,mean_EGF_mo,est_fc_decay_EGF_stack,est_decay_EGF_stack,np.abs(ratio_cut),Freq_fft_cut,ro,alpha,beta)
		################################
		#### Analysis 4: Fit a Brune EGF that has moment of average EGF and stress drop of 1 MPa
		################################	
		### Create Representative Brune model
		brune_freq = sf.bruneModMoFixed(Freq_fft,sf.brune_fc(beta,sf.brune_radii(mean_EGF_mo,1),k),mean_EGF_mo)
		### Main to Brune EGF ratio
		ratio_brune = np.abs(STF_main_FFT)/brune_freq
		freq_smooth_Brune,amp_smooth_brune = sf.resampSpec(Freq_fft[1:],ratio_brune[1:],log_samp) ## Resample spectra
		### Assume fixed decays of 2
		initialGuess=[sf.brune_fc(beta,sf.brune_radii(row['moment_tot'],1),k),sf.brune_fc(beta,sf.brune_radii(mean_EGF_mo,1),k)]## Provide initial guess assume Stress Drop = 1 MPa for initial guess		
		outvalMoFixed = scipy.optimize.least_squares(sf.ratio_bruneModInvMoFixed,initialGuess,bounds=(.00001,np.inf),args=(amp_smooth_brune,freq_smooth_Brune,row['moment_tot'],mean_EGF_mo))
		est_fc_MoFixed_Main_Brune = outvalMoFixed.x[0]
		est_fc_MoFixed_EGF_Brune = outvalMoFixed.x[1]
		sigma_fc_est_Mofix_Brune = sf.stressCircfc(est_fc_MoFixed_Main_Brune,row['moment_tot'],k,beta)
		### Assume variable decays	
		initialGuess=[sf.brune_fc(beta,sf.brune_radii(row['moment_tot'],1),k),2,sf.brune_fc(beta,sf.brune_radii(mean_EGF_mo,1),k),2]## Provide initial guess assume Stress Drop = 1 MPa and decay = 2		
		outval_decay= scipy.optimize.least_squares(sf.ratio_bruneModInvMoFixedslopeFree,initialGuess,bounds=(.00001,np.inf),args=(amp_smooth_brune,freq_smooth_Brune,row['moment_tot'],mean_EGF_mo))
		est_fc_decay_Main_Brune = outval_decay.x[0]
		est_decay_Main_Brune = outval_decay.x[1]
		est_fc_decay_EGF_Brune = outval_decay.x[2]
		est_decay_EGF_Brune = outval_decay.x[3]	
		sigma_fc_est_Mofix_decay_Brune = sf.stressCircfc(est_fc_decay_Main_Brune,row['moment_tot'],k,beta)
		################################
		#### Take means from Analysis 1 and 2
		################################
		est_fc_MoFixed_Main_mean = np.mean(egf_table[egf_table['main_eq_id']==row['eq_id']].est_fc_MoFixed_Main)
		est_fc_decay_Main_mean = np.mean(egf_table[egf_table['main_eq_id']==row['eq_id']].est_fc_decay_Main)
		est_decay_Main_mean = np.mean(egf_table[egf_table['main_eq_id']==row['eq_id']].est_decay_Main)
# 		T_durest_mean = np.mean(egf_table[egf_table['main_eq_id']==row['eq_id']].T_durest)
# 		BRE_T_time_mean = np.mean(egf_table[egf_table['main_eq_id']==row['eq_id']].BRE_T_time)
		stress_fc_MoFixed_mean = sf.stressCircfc(est_fc_MoFixed_Main_mean,row['moment_tot'],k,beta)
		stress_fc_decay_mean = sf.stressCircfc(est_fc_decay_Main_mean,row['moment_tot'],k,beta)
# 		stress_T_durest_mean = sf.stressCircSTF(T_durest_mean,row['moment_tot'],k,beta,cfac)
		################################ Append values to main table
		main_table.loc[len(main_table.index)]=[row['eq_id'],est_fc_MoFixed_Main_stack,est_fc_MoFixed_EGF_stack,
					sigma_fc_est_Mofix_stack,est_fc_decay_Main_stack,est_decay_Main_stack,
					est_fc_decay_EGF_stack,est_decay_EGF_stack,sigma_fc_est_Mofix_decay_stack,
					est_fc_MoFixed_Main_Brune,est_fc_MoFixed_EGF_Brune,sigma_fc_est_Mofix_Brune,
					est_fc_decay_Main_Brune,est_decay_Main_Brune,est_fc_decay_EGF_Brune,
					est_decay_EGF_Brune,sigma_fc_est_Mofix_decay_Brune,mean_EGF_mo,
					est_fc_MoFixed_Main_mean,est_fc_decay_Main_mean,est_decay_Main_mean,
					0,0,stress_fc_MoFixed_mean,stress_fc_decay_mean,
					0,BRE_rat_stack_2_full,BRE_rat_stack_2_cut,BRE_rat_stack_n_full,BRE_rat_stack_n_cut]
		############ ############ ############ ############ ############ ############ ############ ############ ############ ############ 
		############ More plotting code
		if plot_flag == 'Y':
			##### Plot the mean and Brune egfs
			est_ratio = sf.ratio_bruneModMoFixed(freq_smooth,[est_fc_MoFixed_Main_stack,est_fc_MoFixed_EGF_stack],row['moment_tot'],mean_EGF_mo)
			est_ratio_decay = sf.ratio_bruneModMoFixedslopeFree(freq_smooth,[est_fc_decay_Main_stack,est_decay_Main_stack,est_fc_decay_EGF_stack,est_decay_EGF_stack],row['moment_tot'],mean_EGF_mo)
			est_ratio_brune = sf.ratio_bruneModMoFixed(freq_smooth,[est_fc_MoFixed_Main_Brune,est_fc_MoFixed_EGF_Brune],row['moment_tot'],mean_EGF_mo)
			est_ratio_brune_decay = sf.ratio_bruneModMoFixedslopeFree(freq_smooth,[est_fc_decay_Main_Brune,est_decay_Main_Brune,est_fc_decay_EGF_Brune,est_decay_EGF_Brune],row['moment_tot'],mean_EGF_mo)
			ax['l'].loglog(freq_smooth,amp_smooth_stack,color='Blue',alpha=.2)
			ax['l'].axvline(est_fc_MoFixed_Main_stack,color='Blue',ls='--')
# 			ax['l'].loglog(freq_smooth,amp_smooth_brune,color='Red',alpha=.2)
			ax['l'].loglog(freq_smooth,est_ratio,color='Blue')
# 			ax['l'].loglog(freq_smooth,est_ratio_decay,color='Blue',ls='--')
# 			ax['l'].loglog(freq_smooth,est_ratio_brune,color='Red')
# 			ax['l'].loglog(freq_smooth,est_ratio_brune_decay,color='Red',ls='--')
			ratio_txt = "Mean EGF "+ "\n" +r"$f_{{c2}}$ {:.3f} Hz".format(est_fc_MoFixed_Main_stack)+ "\n" +r"$\Delta\sigma_{{fc2}}$ {:.1f} MPa".format(sigma_fc_est_Mofix_stack)#+ "\n" +r"$f_{{cn}}$ {:.3f} Hz".format(est_fc_decay_Main_stack)+ "\n" +r"$\Delta\sigma_{{fcn}}$ {:.1f} MPa".format(sigma_fc_est_Mofix_decay_stack) + "\n"+ r"$n_{{Main}}$: {:.1f}".format(est_decay_Main_stack) + "\n"+ r"$n_{{EGF}}$: {:.1f}".format(est_decay_EGF_stack)				
			ax['l'].text(.01,.01,ratio_txt,ha='left',va='bottom',transform=ax['l'].transAxes,color='Blue')
			ratio_txt = "Brune EGF "+ "\n" +r"$f_{{c2}}$ {:.3f} Hz".format(est_fc_MoFixed_Main_Brune)+ "\n" +r"$\Delta\sigma_{{fc2}}$ {:.1f} MPa".format(sigma_fc_est_Mofix_Brune)#+ "\n" +r"$f_{{cn}}$ {:.3f} Hz".format(est_fc_decay_Main_Brune)+ "\n" +r"$\Delta\sigma_{{fcn}}$ {:.1f} MPa".format(sigma_fc_est_Mofix_decay_Brune) + "\n"+ r"$n_{{Main}}$: {:.1f}".format(est_decay_Main_Brune) + "\n"+ r"$n_{{EGF}}$: {:.1f}".format(est_decay_EGF_Brune)				
# 			ax['l'].text(.21,.01,ratio_txt,ha='left',va='bottom',transform=ax['l'].transAxes,color='Red')			

			ax['a'].set_xlabel('Time (s)')
			ax['a'].set_ylabel('Moment-Rate (N-m/s)')
			ax['b'].set_xlabel('Time (s)')
			ax['b'].set_ylabel('Moment-Rate (N-m/s)')			
			ax['c'].set_xlabel('Time (s)')
			ax['c'].set_ylabel('Moment-Rate (N-m/s)')
			ax['d'].set_xlabel('Time (s)')
			ax['d'].set_ylabel('Moment-Rate (N-m/s)')		
			ax['f'].set_xlabel('Frequency (Hz)')
			ax['f'].set_ylabel('Moment (N-m)')		
# 			ax['i'].set_xlabel('Time (s)')
# 			ax['i'].set_ylabel('Rel. Moment-Rate')	
			ax['k'].set_xlabel('Frequency (Hz)')
			ax['k'].set_ylabel('Rel. Moment')	
			ax['l'].set_xlabel('Frequency (Hz)')
			ax['l'].set_ylabel('Rel. Moment')	
			##### Save Plot
			figfile = plotpath +"/Decon_"+row['eq_id']+".pdf"
# 			fig.tight_layout()
			plt.savefig(figfile)
			plt.close()
			plot_flag = 'N' # Reset the flag
			plot_count +=1 # Track th number of plots
		############ ############ ############ ############ ############ ############ ############ ############ ############ ############ 

			
			
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





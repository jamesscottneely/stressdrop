#!/usr/bin/env python3  

#### This script plots stress drop histograms
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.signal as sig
import stress_func as sf
import pandas as pd
from datetime import datetime
import os
import itertools
import similaritymeasures as sm
import scipy.integrate as integrate
import scipy.stats as stats
import shutil


#### Randomly sample moments
def Mom_random(beta_m,Mt,Mm,num_events,rng):
	moment_Main = Mm*1.5 # initialize array
	while moment_Main > Mm: # Make sure mo of main earthquake below max value otherwise repeat
		moment_Main = stats.pareto.rvs(b=beta_m,scale=Mt,size=1,random_state=rng)	
	#### Randomly select weights of sub-events
	weight_sub = rng.uniform(0,1,size=num_events) # First determine weights based uniform random distribution
	weight_sub = weight_sub/np.sum(weight_sub) #   normalize so sum = 1
	#### Convert Weights to seismic moments
	moment_sub = weight_sub*moment_Main
	return moment_sub
	


#### Set constant variables
##### Variables of interests
s_type = 'Vary' # "Vary" or "Fix" for stress drops of synthetic earthquakes
s_mean_in = 0 # Mean of the overall stress drop distribution in log10 units 
s_sd_in = .5 # SD of the overall stress drop distribution in log10 units
beta = 3600. # s-wave velocity m/s
alpha = beta*np.sqrt(3) # p-wave velocity m/s
rup_vel = .9*beta # rupture velocity
theta = np.pi/2 # Angle (radians) relative to fault 
ro = 2800 # density kg/m^3
mu = .5*1e9 # shear modulus
samp_rate = .05 # Set the sampling rate for synthetics
log_samp = 0.025 # Log sampling for fitting spectral model
t = np.arange(0,1000,samp_rate) # Array of time values
k = 0.37 # Shape factor brune calculation
c = 1 # Constant for Fc to T conversion
maxHz = 0.5*0.5*1/samp_rate # Maximum fitting range for brune model
seed_val  = 83714371 # for randomly generating numbers
rng = np.random.default_rng(seed=seed_val) # Set the seed value for the random number generator
numEqs = 5000 #  number of random earthquakes to generate
minEvents = 2 # min number of sub-events
maxEvents = 5 # Max number of sub-events
percent = .1 # percent of maxing moment rate for duration fitting
beta_m = 2./3 # Value for Magnitude distribution
MwMin = 5 # Minimum Mag
MwMax = 8# Maximim Mag
Mt = sf.mw2mo(MwMin) # Minimum Moment
Mm =  sf.mw2mo(MwMax) # Maximum Moment
fm = 1/(samp_rate*.5) # Nyquist
cval = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5)))) # constant parameter
mod_Type = "Brune" # Brune or Sato (Sato & Hirasawa)
dataout = pd.DataFrame(columns=['eq_id','num_events','stress_i','moment_i','mw_i','radii_i','t_i',
	'fc_i','rup_i','moment_tot','sigma_tot_moment','sigma_tot_area','mw_tot','T_dur',
	'sigma_T_est','BRE_T_time','TDur_Ends','rad_energy_time','eff_T',
	'est_fc_MoFixed','sigma_fc_est_Mofix','est_fc_freeslope',
	'est_fs_freeslope','sigma_fc_est_freeslope','est_Mo_freeMo','est_fc_freeMo',
	'sigma_fc_est_freeMo','rad_energy_freq','eff_freq','fcratio_fc_freq','est_Mo_freeAll',
	'est_fc_freeAll','est_fs_freeAll','BRE_Freq_2','BRE_Freq_n'])


	
	
	
### Generat run ID
nowt = datetime.now()
runID = nowt.strftime('%Y%m%d%s')
### Set path
dstr  = mod_Type+"_{:.2f}SD_".format(s_sd_in) + "{:.0f}HZ".format(1/samp_rate) #
figdir = "ENTER DIRECTORY PATH to SAVE FIGURES" + dstr
if os.path.exists(figdir):
	pass
else:
	os.mkdir(figdir)
	
path = 'ENTER DIRECTORY PATH to SAVE DATA' + dstr
if os.path.exists(path):
	pass
else:
	os.mkdir(path)
	
STFPath = path + '/STFs'
if os.path.exists(STFPath):
	shutil.rmtree(STFPath)
	os.mkdir(STFPath)
else:
	os.mkdir(STFPath)

	
for eq in range(numEqs):
	print(eq)
	eq_id = runID + "_{:04d}".format(eq)
	###################################### Create Pulse
	###################################### 
	###################################### 
	#### Step 1: Number of events in each earthquake (Uniform distribution)
	num_events = rng.integers(low=minEvents, high=maxEvents, size=1,endpoint=True)
	#### Step 2: Stress drops of each sub-event 
	if s_type=='Fix':
		stress_sub = np.repeat(10**s_mean_in,num_events) # Set all sub-events to same stress dop
	elif s_type=='Vary':
		stress_sub = 10**rng.normal(s_mean_in,s_sd_in,size=num_events) # Generate random stress drops i from lognormal distribution. 10** converts to MPa units.
	#### Step 3: Randomly select moments from Pareto distribution (see Kagan 2002 eq 4)
	moment_sub = Mom_random(beta_m,Mt,Mm,num_events,rng)
	seis_mom_val = np.sum(moment_sub)
	#### Step 4: Convert Seismic moments to Mw
	mw_sub = sf.mo2mw(moment_sub)
	mw = sf.mo2mw(seis_mom_val)
	
	
	##### The following applies to Brune only
	if mod_Type == "Brune":
		### Step 5: Calculate the radii of the sub-events
		radii_sub = sf.brune_radii(moment_sub,stress_sub)
		area_sub = np.pi*np.square(radii_sub)
		### Step 6: Calculate the corner frequencies of the sub-events
		fc_sub = sf.brune_fc(beta,radii_sub,k)
		### Step 7: Calculate the ruptur duration of the sub-events
		Tdur_sub = 1/fc_sub
		#### Step 8: Randomly generate time between sub-event
		start_sub = np.zeros(num_events) # Initialize the sub-event start time array
		s_val = 0 # Start value for pulse start
		e_val = np.max(Tdur_sub) # End value for pulse start
		for sub in range(int(num_events)):
			if sub == 0: # If largest sub-event
				start_sub[sub] = 0 # Time shift is always zero for largest pulse
			else:
				s_val = np.min(start_sub)-Tdur_sub[sub] # Minimum possible start value to ensure overlap: min(start_time) - T_dur next pulse
				e_val = np.max(start_sub[:sub]+Tdur_sub[:sub]) # Maximum possible start value to ensure overlap
				start_sub[sub] = rng.uniform(s_val,e_val) # Start time of pulse
		#### Step 9: Creat the brune pulses
		pulse_sub = sf.brune_time_moment(np.atleast_2d(moment_sub).T,np.atleast_2d(fc_sub).T,np.atleast_2d(start_sub).T+np.mean(t),np.tile(t,(int(num_events),1))) # +50 on the start_sub ensures that it is shifted properly away from 0. Need to change shape for calculation
		pulse_sub[pulse_sub<0] = 0 # Remove negative values

	elif mod_Type == "Sato":
		radii_sub = sf.sato_radii(moment_sub,stress_sub*1e6)#
		area_sub = (radii_sub*np.pi)**2
		fc_sub = sf.brune_fc(beta,radii_sub*1e-3,k) # Approximation of corner frequency
		Tdur_sub = 1/fc_sub
		#### Randomly generate time between sub-event
		start_sub = np.zeros(num_events) # Initialize the sub-event start time array
		s_val = 0 # Start value for pulse start
		e_val = np.max(Tdur_sub) # End value for pulse start
		for sub in range(int(num_events)):
			if sub == 0: # If largest sub-event
				start_sub[sub] = 0 # Time shift is always zero for largest pulse
			else:
				s_val = np.min(start_sub)-Tdur_sub[sub] # Minimum possible start value to ensure overlap: min(start_time) - T_dur next pulse
				e_val = np.max(start_sub[:sub]+Tdur_sub[:sub]) # Maximum possible start value to ensure overlap
				start_sub[sub] = rng.uniform(s_val,e_val) # Start time of pulse
		#### Step 9: Creat the brune pulses
# 		pulse_sub = sf.brune_time_moment(np.atleast_2d(moment_sub).T,np.atleast_2d(fc_sub).T,np.atleast_2d(start_sub).T+np.mean(t),np.tile(t,(int(num_events),1))) # +50 on the start_sub ensures that it is shifted properly away from 0. Need to change shape for calculation
		pulse_sub = np.zeros((num_events[0],len(t)))
		idx_sub = 0
		for sub in range(num_events[0]):
			pulse_sub[idx_sub,:] = sf.sh_1973(radii_sub[idx_sub],rup_vel,stress_sub[idx_sub]*1e6,theta,beta,t,start_sub[idx_sub]+np.mean(t))
			idx_sub+=1	
	
	#### Step 10: Sum the pulses for the final pulse
	pulse_sum = np.sum(pulse_sub,axis=0)	
	### Weighted Averages of stress drops
	sigma_Mo = np.sum((moment_sub/np.sum(moment_sub))*stress_sub) # Moment weight average
	sigma_area = np.sum((area_sub/np.sum(area_sub))*stress_sub) # Area weighted average
	###################################### Measure Pulse
	###################################### 
	###################################### 	
	
	#################### Time Domain
	##### Measure rupture duration start known
# 	T_dur,epoints = sf.rupture_Duration(pulse_sum,samp_rate,percent)
# 	epoint_start = [ n for n,i in enumerate(pulse_sum) if i>0][0]
# 	T_dur = T_dur - t[epoint_start]
	###### Measure duration start unknown
	T_dur,epoints = sf.rupture_Duration_StartUnk(pulse_sum,samp_rate,percent)

	####  time estimate of stress drop
	sigma_T_est = sf.stressCircSTF(T_dur,seis_mom_val,k,beta,c)
	#### generate Brune in Time with same duration
	Brune_T_time = sf.brune_time_moment(seis_mom_val,c/T_dur,t[epoints[0]],t)
	Brune_T_time[Brune_T_time<0] = 0 # trim the initial part of Brune
	#### Calculate Brune Radiated energy
	BRE_T_time = sf.rough_fun(Brune_T_time,pulse_sum,samp_rate)
	#### Calculate radiated energy in Time domain
	rad_energy_Time = sf.radE_STF(seis_mom_val,pulse_sum,samp_rate,ro,alpha,beta)

	#################### Frequency Domain
	##### Trim the STF so length relative to rupture duration
	t_trim = t#sf.trim_STF(pulse_sum,t,[epoint_start,epoints],samp_rate) Don't trim in this case to avoid cutting the signal
	STF_trim = pulse_sum
	#### Perform FT
	freq,amplitude,FT_complex = sf.sig_process(t_trim,STF_trim,maxHz) # Process the signal to spectra 
	#### Perform FFT for energy calculation
	freq_e,amplitude_e,FT_complex_e = sf.sig_process_energy(t_trim,STF_trim,maxHz)
	#### Smooth spectra
	freq_smooth,amp_smooth = sf.resampSpec(freq,amplitude,log_samp) ## Resample spectra
	### Fit Fc only
	initialGuess=[1/T_dur]## Provide initial guess
	outvalMoFixed = optimize.least_squares(sf.bruneModInvMoFixed,initialGuess,bounds=(.0001,np.inf),args=(amp_smooth,freq_smooth,seis_mom_val))
	est_fc_MoFixed = outvalMoFixed.x[0]
	sigma_fc_est_Mofix = sf.stressCircfc(est_fc_MoFixed,seis_mom_val,k,beta)		
	### Fit fc and slope	
	initialGuess=[1/T_dur,2]## Provide initial guess start
	outval = optimize.least_squares(sf.bruneModInvMoFixedslopeFree,initialGuess,bounds=((.0001,.01),(np.inf,np.inf)),args=(amp_smooth,freq_smooth,seis_mom_val)) ## making the actual fit
	est_fc_freeslope = outval.x[0]
	est_fs_freeslope =  outval.x[1]
	sigma_fc_est_freeslope = sf.stressCircfc(est_fc_freeslope,seis_mom_val,k,beta)
	### Fit Mo and fc
	initialGuess=[seis_mom_val,1/T_dur]## Provide initial guess start
	outval = optimize.least_squares(sf.bruneModInv,initialGuess,bounds=((-np.inf,.0001),(np.inf,np.inf)),args=(amp_smooth,freq_smooth)) ## making the actual fit
	est_Mo_freeMo = outval.x[0]
	est_fc_freeMo =  outval.x[1]
	sigma_fc_est_freeMo = sf.stressCircfc(est_fc_freeMo,est_Mo_freeMo,k,beta)
	### Fit Mo,fc,slope
	initialGuess=[seis_mom_val,1/T_dur,2]## Provide initial guess start
	outval = optimize.least_squares(sf.bruneModInv_AllFree,initialGuess,bounds=((-np.inf,.0001,.01),(np.inf,np.inf,np.inf)),args=(amp_smooth,freq_smooth)) ## making the actual fit
	est_Mo_freeAll = outval.x[0]
	est_fc_freeAll =  outval.x[1]
	est_fs_freeAll =  outval.x[1]
	#### Radiated Energy
	rad_energy_freq = sf.radE_STF_freq(amplitude_e,freq_e,ro,alpha,beta)
	bestBrune = sf.bruneModMoFixed(freq_e,[est_fc_MoFixed],seis_mom_val)
	best_n = sf.bruneModMoFixedfslopeFree(freq_e,[est_fc_freeslope,est_fs_freeslope],seis_mom_val)
	rad_energy_n = sf.radE_STF_freq(best_n,freq_e,ro,alpha,beta)
	rad_energy_2 = sf.radE_STF_freq(bestBrune,freq_e,ro,alpha,beta)
	

	###################################### Save data
	###################################### 
	###################################### 	
	#### Defined information
	dataout.at[eq,'eq_id'] = eq_id
	dataout.at[eq,'moment_tot'] = seis_mom_val	
	dataout.at[eq,'sigma_tot_moment'] = sigma_Mo	
	dataout.at[eq,'sigma_tot_area'] = sigma_area	
	dataout.at[eq,'mw_tot'] = mw	
	dataout.at[eq,'num_events'] = num_events
	dataout.at[eq,'stress_i'] = stress_sub.tolist()
	dataout.at[eq,'moment_i'] = moment_sub.tolist()
	dataout.at[eq,'mw_i'] = mw_sub.tolist()
	dataout.at[eq,'radii_i'] = radii_sub.tolist()
	dataout.at[eq,'t_i'] = start_sub.tolist()
	dataout.at[eq,'fc_i'] = fc_sub.tolist()	
	dataout.at[eq,'rup_i'] = Tdur_sub.tolist()	

	#### Estimated information
	dataout.at[eq,'T_dur'] = T_dur
	dataout.at[eq,'TDur_Ends'] =epoints
	dataout.at[eq,'sigma_T_est'] = sigma_T_est
	dataout.at[eq,'est_fc_mofixed'] = est_fc_MoFixed
	dataout.at[eq,'BRE_T_time'] = BRE_T_time
	dataout.at[eq,'rad_energy_time'] = rad_energy_Time
	dataout.at[eq,'eff_T'] = rad_energy_Time/seis_mom_val
	dataout.at[eq,'est_fc_MoFixed']=est_fc_MoFixed
	dataout.at[eq,'sigma_fc_est_Mofix']=sigma_fc_est_Mofix
	dataout.at[eq,'est_fc_freeslope']=est_fc_freeslope
	dataout.at[eq,'est_fs_freeslope']=est_fs_freeslope
	dataout.at[eq,'sigma_fc_est_freeslope']=sigma_fc_est_freeslope
	dataout.at[eq,'est_Mo_freeMo']=est_Mo_freeMo
	dataout.at[eq,'est_fc_freeMo']=est_fc_freeMo
	dataout.at[eq,'sigma_fc_est_freeMo']=sigma_fc_est_freeMo
	dataout.at[eq,'rad_energy_freq']=rad_energy_freq
	dataout.at[eq,'eff_freq'] = rad_energy_freq/seis_mom_val
	dataout.at[eq,'fcratio_fc_freq'] = est_fc_MoFixed/est_fc_freeslope		 
	dataout.at[eq,'est_Mo_freeAll'] = est_Mo_freeAll
	dataout.at[eq,'est_fc_freeAll'] = est_fc_freeAll
	dataout.at[eq,'est_fs_freeAll'] = est_fs_freeAll
	dataout.at[eq,'BRE_Freq_2'] = rad_energy_freq/rad_energy_2
	dataout.at[eq,'BRE_Freq_n'] = rad_energy_freq/rad_energy_n
	
	### Save STF Time Series 
	stffile = STFPath +"/"+ eq_id + ".txt"
	outdic = {'time':np.arange(len(STF_trim))*samp_rate,'STF':STF_trim}
	stf_out = pd.DataFrame(data = outdic) 
	stf_out.to_csv(stffile,sep=',',index=False)

	######### ######### ######### ######### ######### ######### ######### 
	######### Aside: Plot a few examples
	if eq < 25:
# 		plt.rcParams.update({'font.size': 18})
# 		Brune_T_time = sf.brune_time_moment(seis_mom_val,1/T_dur,t[epoints[0]],t) # Start it at the end points
# 		Brune_T_time[Brune_T_time<0] = 0
# 		t_trim, STF_trim = sf.trim_STF(pulse_sum,t,[epoint_start,epoints],samp_rate)
# 		t_trim,Brune_T_time =  sf.trim_STF(Brune_T_time,t,[epoint_start,epoints],samp_rate)
		fig, [ax1,ax2] = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
		#### Plot STF
		ax1.plot(t_trim,STF_trim,linewidth=3,color='black',label='Complex') # Plot sub-event
# 		ax1.plot(np.tile(t_trim,(int(num_events),1)).T,pulse_sub.T,linewidth=1) # Plot sub-event
		ax1.plot(t_trim,Brune_T_time,linewidth=1,color='red',label='Model (Time)')
		hlevel = ax1.get_ylim()[0]/2
# 		T_dur_plot,epoints_plot = sf.rupture_Duration(STF_trim,samp_rate,percent)
# 		epoint_start = [ n for n,i in enumerate(STF_trim) if i>0][0]
# 		T_dur = T_dur - t_trim[epoint_start]
		ax1.hlines(hlevel,t_trim[epoints[0]],t_trim[epoints[1]],lw=1,color='red',ls='--')
		tdur_str = "{:.1f}".format(T_dur) + " s"
		ax1.text((t_trim[epoints[0]]+t_trim[epoints[1]])/2,hlevel,tdur_str,va='bottom',ha='center')

		### Plot Spectra
		bestBrune = sf.bruneModMoFixed(freq,[est_fc_MoFixed],seis_mom_val)
		ax2.loglog(freq,amplitude,color='black',label='Complex')
		ax2.loglog(freq,bestBrune,color='orange',lw=1,label = 'Model (Freq.)')
		ax2.axvline(est_fc_MoFixed,ls='--',color='orange')


		fc_str = "{:.2f}".format(est_fc_MoFixed) + " Hz"
		ax2.text(est_fc_MoFixed,seis_mom_val*.01,fc_str,va='bottom',ha='right',rotation=90)
		
		#### Add annotations
		STF_str = 	r'$\Delta\sigma_{\overline{Area}}$: ' +"{:.1f}".format(sigma_area) + ' MPa' +"\n"+ r'$\Delta\sigma_{\overline{Mo}}$: '+ "{:.1f}".format(sigma_Mo) + " MPa"+"\n" +r'$\widehat{\Delta\sigma_{T}}$: '+ "{:.1f}".format(sigma_T_est) + " MPa"+"\n" + 'BRE: ' + "{:.1f}".format(BRE_T_time)
		ax1.text(.99,.99,STF_str,ha='right',va='top',transform=ax1.transAxes)			
		
		spec_str = r'$\widehat{\Delta\sigma_{fc}}$: '+ "{:.1f}".format(sigma_fc_est_Mofix) + " MPa"+"\n"  + 'Decay: ' + "{:.1f}".format(est_fs_freeslope) + "\n" + "Scaled E: " + "{:.2E}".format(rad_energy_freq/seis_mom_val)
		ax2.text(.99,.99,spec_str,ha='right',va='top',transform=ax2.transAxes)			
		ax1.set_title('STF')
		ax1.set_ylabel('Moment Rate (N-m/s)')
		ax1.set_xlabel('Time (s)')
		ax1.set_xlim([t_trim[epoints[0]]-T_dur*.5,t_trim[epoints[1]]+T_dur*.5])


		ax2.set_title('STF Spectral Amplitude')
		ax2.set_ylabel('Moment (N-m)')
		ax2.set_xlabel('Frequency (Hz)')
		
# 		ax1.legend(loc=7)
# 		ax1a.legend(loc=7)
# 		ax2.legend(loc=7)
		#### Save file	
		figfile = figdir+ "/fig_synth_seed_" +str("{:.0f}".format(seed_val)) + "_idx_" +str(eq) + "_mod"+mod_Type +".pdf"
		fig.tight_layout()
		plt.savefig(figfile)
		plt.close()

	eq+=1

	######### ######### ######### ######### ######### ######### ######### 



#### Save meta data file


datafile =  path + '/eqFile.txt'
dataout.to_csv(datafile,sep='|',index=False)
### Save general inputs
inputout = pd.DataFrame(columns=['runID','maxEvents','s_mean_in','s_sd_in','num_events','beta','mu','samp_rate','log_samp','k','c',
	'cutoff_per','band','beta_m','MwMin','MwMax','maxHz','seedVal'])
inputout.at[0,'runID'] = runID
inputout.at[0,'maxEvents']= maxEvents
inputout.at[0,'s_mean_in']= s_mean_in
inputout.at[0,'s_sd_in']= s_sd_in
inputout.at[0,'num_events'] = num_events
inputout.at[0,'beta'] = beta
inputout.at[0,'mu'] = mu
inputout.at[0,'samp_rate'] = samp_rate
inputout.at[0,'log_samp'] = log_samp
inputout.at[0,'k'] = k
inputout.at[0,'c'] = c
inputout.at[0,'cutoff_per'] = percent
inputout.at[0,'band'] = band
inputout.at[0,'beta_m'] = beta_m 
inputout.at[0,'MwMin'] = MwMin
inputout.at[0,'MwMax'] = MwMax
inputout.at[0,'maxHz'] = maxHz
inputout.at[0,'seedVal'] = seed_val
inputfile =  path + '/inputVals.txt'
inputout.to_csv(inputfile,sep='|',index=False)


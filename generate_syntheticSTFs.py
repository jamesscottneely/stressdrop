#!/usr/bin/env python3  

#### This script plots stress drop histograms
import numpy as np
import stress_func as sf
import pandas as pd
from datetime import datetime
import os
import scipy.stats as stats
import shutil


#### Randomly sample moments
def Mom_random(beta_m,Mt,Mm,num_events,rng,MwMin,MwMax,sampType):
    moment_Main = Mm*1.5 # initialize array
    if sampType == 'GR':
        while moment_Main > Mm: # Make sure mo of main earthquake below max value otherwise repeat
            moment_Main = stats.pareto.rvs(b=beta_m,scale=Mt,size=1,random_state=rng)
    elif sampType == 'Uniform':
        mw_Main = stats.uniform.rvs(loc=MwMin,scale=MwMax-MwMin,size=1,random_state=rng)
        moment_Main = sf.mw2mo(mw_Main)
    #### Randomly select weights of sub-events
    weight_sub = rng.uniform(0,1,size=num_events) # First determine weights based uniform random distribution
    weight_sub = weight_sub/np.sum(weight_sub) #   normalize so sum = 1
    ##### Convert Weights to seismic moments
    moment_sub = weight_sub*moment_Main
    return moment_sub
	


#### Set constant variables
##### Variables of interests
sampType = 'Uniform'
s_type = 'Vary' # "Vary" or "Fix" for stress drops of synthetic earthquakes
s_mean_in = 0 # Mean of the overall stress drop distribution in log10 units 
s_sd_in = .5 # SD of the overall stress drop distribution in log10 units
beta = 3600. # s-wave velocity m/s
alpha = beta*np.sqrt(3) # p-wave velocity m/s
rup_vel = .9*beta # rupture velocity
theta = np.pi/2 # Angle (radians) relative to fault 
ro = 2800 # density kg/m^3
mu = .5*1e9 # shear modulus
samp_rate = .01 # Set the sampling rate for synthetics
t = np.arange(0,1000,samp_rate) # Array of time values
k = 0.37 # Shape factor brune calculation
c = 1 # Constant for Fc to T conversion
seed_val  = 83714371 # for randomly generating numbers
rng = np.random.default_rng(seed=seed_val) # Set the seed value for the random number generator
numEqs = 5000 #  number of random earthquakes to generate
minEvents = 2 # min number of sub-events
maxEvents = 5 # Max number of sub-events
beta_m = 2./3 # Value for Magnitude distribution
MwMin = 4 # Minimum Mag
MwMax = 8# Maximim Mag
Mt = sf.mw2mo(MwMin) # Minimum Moment
Mm =  sf.mw2mo(MwMax) # Maximum Moment
fm = 1/(samp_rate*.5) # Nyquist
cval = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5)))) # constant parameter
mod_Type = "Brune" # Brune or Sato (Sato & Hirasawa)
dataout = pd.DataFrame(columns=['eq_id','num_events','stress_i','moment_i','mw_i','radii_i','t_i',
	'fc_i','rup_i','moment_tot','sigma_tot_moment','sigma_tot_area','mw_tot'])

### Generat run ID
nowt = datetime.now()
runID = nowt.strftime('%Y%m%d%s')
### Set path
dstr  = mod_Type+"_{:.2f}SD_".format(s_sd_in) + "{:.0f}HZ".format(1/samp_rate) +"_"+sampType#

path = '/Users/jamesneely/Documents/NSF/StressDrop_Bands/' + dstr
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
	moment_sub = Mom_random(beta_m,Mt,Mm,num_events,rng,MwMin,MwMax,sampType)
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


	### Save STF Time Series 
	stffile = STFPath +"/"+ eq_id + ".txt"
	outdic = {'time':t,'STF':pulse_sum}
	stf_out = pd.DataFrame(data = outdic) 
	stf_out.to_csv(stffile,sep=',',index=False)

	eq+=1

	######### ######### ######### ######### ######### ######### ######### 



#### Save meta data file


datafile =  path + '/eqFile.txt'
dataout.to_csv(datafile,sep='|',index=False)
### Save general inputs
inputout = pd.DataFrame(columns=['runID','sampType','maxEvents','s_mean_in','s_sd_in','num_events','beta','mu','samp_rate','k','c',
	'beta_m','MwMin','MwMax','seedVal'])
inputout.at[0,'runID'] = runID
inputout.at[0,'sampType'] = sampType
inputout.at[0,'maxEvents']= maxEvents
inputout.at[0,'s_mean_in']= s_mean_in
inputout.at[0,'s_sd_in']= s_sd_in
inputout.at[0,'num_events'] = num_events
inputout.at[0,'beta'] = beta
inputout.at[0,'mu'] = mu
inputout.at[0,'samp_rate'] = samp_rate
inputout.at[0,'k'] = k
inputout.at[0,'c'] = c
inputout.at[0,'beta_m'] = beta_m 
inputout.at[0,'MwMin'] = MwMin
inputout.at[0,'MwMax'] = MwMax
inputout.at[0,'seedVal'] = seed_val
inputfile =  path + '/inputVals.txt'
inputout.to_csv(inputfile,sep='|',index=False)


#!/usr/bin/env python3  

#### This document contains the functions necessary for this analysis
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.signal as sig
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.special as spec
import statsmodels.api as sm



##### Generate the displacement time series for Brune source	
def brune_time_disp(sigma,t_shift,r,beta,mu,t):
	b = 2.33*beta/r # Calculate angular corner frequency
	disp = (sigma*beta/mu)*(t-t_shift)*np.exp(-b*(t-t_shift)) ### Modified from 18.28 Principles of Seimsmology
	return disp

######### Following equations from ide and beroza 2001
# def Brune_Vel(Mo,f,fc): # EQ 1
# 	return Mo*f/(1+(f/fc))**2
# 
# def Brune_E(Mo,fc): # EQ 2
# 	return (1/4)*np.pi*(Mo**2)*(fc**3)
# 	
# def F_fun(fm,fc):  # Eq. 3
# 	F = (-fm/fc)/((1+(fm/fc))**2)+np.arctan2(fm,fc)
# 	return F
# 	
# def Brune_E_Obs(Mo,fm,fc): #EQ 4
# 	F = F_fun(fm,fc)
# 	return (1/2)*(Mo**2)*(fc**3)*F
# 	
# def mis_energy(fc,k,beta,fm): # EQ 5
# 	F = F_fun(fm,fc)
# 	R = (2/np.pi)*F # EQ 5
# 	return R
	

##### Generate moment rate function for a brune model	
def brune_time_moment(sigma,t_shift,r,beta,mu,t):
	b = 2.33*beta/r # Calculate angular corner frequency
	Mo = brune_mom(r/1000,sigma/1e6) # division needed to convert back to km and MPa
	moment_rate = Mo*(b**2)*(t-t_shift)*np.exp(-b*(t-t_shift))
	return moment_rate
	
##### Generate moment rate function for a brune model	
def brune_time_moment(Mo,fc,t_shift,t):
	b = 2*np.pi*fc # Calculate angular corner frequency
	moment_rate = Mo*(b**2)*(t-t_shift)*np.exp(-b*(t-t_shift))
	return moment_rate

#### Calculate moment of brune model
def brune_mom(r,sigma):
	# r in km; sigma in MPa
	r_m = r*1000 # Convert r from km to m
	sigma_pa = sigma*1e6 # Convert MPa to Pa
	moment = (16./7.)*sigma_pa*(r_m**3) # Calculate moment in N-m
	return moment
	
#### Calculate radii of brune model given moment and sigma
def brune_radii(mo,sigma):
	# r in km; sigma in MPa
	sigma_pa = sigma*1e6 # Convert MPa to Pa
	r_m = ((7./16.)*mo/(sigma_pa))**(1./3.)  # Calculate radii in m
	return r_m/1000 # convert to km on retuen

### Calculate Brune corner frequency
def brune_fc(beta,r,kappa):
	# beta in m/s. r in km
	fc = kappa*beta/(r*1000)
	return fc
	

def mw2mo(mw):
	# Convert seismic moment (N-m) to moment magnitude
	mo = (10**((mw+10.7)*(3./2.)))/(1e6)
	return mo

def mo2mw(mo):
	# Convert seismic moment (N-m) to moment magnitude
	Mw = (2./3.)*np.log10(mo*1e6)-10.7
	return Mw


##### Divide up stress drops
def sub_events(num_events,tot_area,beta,weight_type,weights="Random",stress="Random",time="Random",**kwargs):
	#### Check if stress drops should be random
	if stress == 'Random':
		rng = kwargs['rng']
		stress_i = 10**rng.normal(kwargs['s_mean'],kwargs['s_sd'],num_events) # Generate random stress drops i from lognormal distribution. 10** converts to MPa units
	else: 
		stress_i = np.array(kwargs['stress_i']) # Values entered manually
	#### Calculate the sub-event weights
	if num_events == 1:
		weight_i = int(1) # Always return weight of 1
	else:
		if weights == 'Random': # Random weights
			weight_i = [] # initialize weights array
			for w in range(num_events-1): # Loop but don't include last value
				weight_i.append(rng.uniform(0,1-np.sum(weight_i))) # Generate random weights (based on fault area)
			weight_i.append(1-np.sum(weight_i)) # Last weight is determined based on sum of remaining values
		else:
			weight_i = kwargs['weight_i'] # Values entered manually
	weight_i = np.array(weight_i) # Convert from list to numpy array
	### Check if the weights correspond to area or seismic moment
	if weight_type == 'Area':
		### Use weights to calculate fault radii for sub-events
		radii_i = np.sqrt(tot_area*weight_i/np.pi) # Returns radii in km
		### Calculate moments of sub-events
		moment_i = brune_mom(radii_i,stress_i)
		#### Calculate total sigma based on area weights
		sigma_tot_area = np.sum(stress_i*weight_i)
		### Calculate total sigma based on moment weights
		weight_mom = moment_i/np.sum(moment_i)
		sigma_tot_moment = np.sum(stress_i*weight_mom)
		weight_area = weight_i
	elif weight_type == 'Moment':
		### Calculate moments based on weights
		moment_i = weight_i/(np.min(weight_i))*kwargs['seis_mom'] # 'seis_mom' is the seismic moment of the smaller event
		### Calculate radii based on moments
		radii_i = brune_radii(moment_i,stress_i)
		### Calculate area weights
		weight_area = (radii_i**2)*np.pi/np.sum((radii_i**2)*np.pi)
		#### Calculate total sigma based on moment weights
		sigma_tot_moment = np.sum(stress_i*weight_i)
		#### Calculate total sigma based on area weights
		sigma_tot_area = np.sum(stress_i*weight_area)
		weight_mom = weight_i

	#### Calculate total moment by summing the components
	moment_tot = np.sum(moment_i)
	#### Overall parameters
	r_tot_est = np.sqrt(tot_area/np.pi) # Estimated radii based on input area
	t_est =  1/brune_fc(beta,r_tot_est) # Estimated rupture duration for that fault radii
	fc_i = brune_fc(beta,radii_i) # Corner frequencies of sub events
	rup_i = 1/brune_fc(beta,radii_i) # rupture durations of sub events

	### Calculate the time shift
	if time == 'Random':
		t_i = [0]
		for t in range(num_events-1): # Loop but don't include last value
				t_i.append(rng.uniform(-t_est/2,t_est/2)) # Generate random time shifts around mean
	else: 
		t_i = kwargs['t_i'] # Values entered manually
	t_i_np = np.array(t_i)+15

	return stress_i,moment_i,radii_i,t_i_np,fc_i,rup_i,weight_mom,weight_area,moment_tot,sigma_tot_moment,sigma_tot_area
	
def sig_process(timeVal,momentRate,maxHz):
	#### Process Function for FFT
	dt = timeVal[-1]-timeVal[-2]
	momentRate_pad = momentRate # now pad before this step #padfunc(momentRate) # Pad time series
	t_pad = timeVal# now pad before this step  np.concatenate((timeVal,(max(timeVal)+dt*(1+np.arange((len(momentRate_pad) - len(timeVal))))))) # Pad time values
	# momentRate_pad = momentRate # don't Pad time series
# 	t_pad = timeVal # don't Pad time values
	#### Perform fft
	N = len(momentRate_pad)
	freq = np.fft.rfftfreq(N, d = dt) ### Find frequency values
	FT_momentRate  = np.fft.rfft(momentRate_pad)*dt #### Peform FFT
	#### Trim the files to  Max N Hz
	freq = freq[freq<=maxHz]
	cutInt = len(freq)
	freq = freq[1:] 
	FT_momentRate = FT_momentRate[1:cutInt]
	#### Analyze spectra
	amplitude_FT = np.abs(FT_momentRate)
	return (freq,amplitude_FT,FT_momentRate)
	
	
def sig_process_energy(timeVal,momentRate,maxHz):
	#### Process Function for FFT
	dt = timeVal[-1]-timeVal[-2]
# 	momentRate_pad = padfunc(momentRate) # Pad time series
# 	t_pad = np.concatenate((timeVal,(max(timeVal)+dt*(1+np.arange((len(momentRate_pad) - len(timeVal))))))) # Pad time values
	#### Perform fft
	N = len(momentRate)
	freq = np.fft.rfftfreq(N, d = dt) ### Find frequency values
	FT_momentRate  = np.fft.rfft(momentRate)*dt #### Peform FFT
	#### Trim the files to  Max N Hz
	freq = freq[freq<=maxHz]
	cutInt = len(freq) 
	FT_momentRate = FT_momentRate[:cutInt]
	#### Analyze spectra
	amplitude_FT = np.abs(FT_momentRate)
	return (freq,amplitude_FT,FT_momentRate)
	
def fft_full(timeVal,momentRate):
	#### Process Function for FFT
	dt = timeVal[-1]-timeVal[-2]
	# momentRate_pad = padfunc(momentRate) # Pad time series
# 	t_pad = np.concatenate((timeVal,(max(timeVal)+dt*(1+np.arange((len(momentRate_pad) - len(timeVal))))))) # Pad time values
# 	#### Perform fft
# 	N = len(momentRate_pad)
	N = len(timeVal)
	freq = np.fft.rfftfreq(N, d = dt) ### Find frequency values
	FT_momentRate  = np.fft.rfft(momentRate)*dt #### Peform FFT
	#### Analyze spectra
	amplitude_FT = np.abs(FT_momentRate)
	return (freq,amplitude_FT,FT_momentRate)



def padfunc(x):
	N = 2*len(x) #
# 	N = int(2**np.ceil(np.log2(2*len(x)))) # Double length, Find next power 2 & corresponding number
	# A0 = np.zeros((N-len(x))) # Create array with zeros
# 	outval = np.concatenate((x,A0))
	outval = np.pad(x, (int((N-len(x))/2), int((N-len(x))/2)+1), 'constant')
	return outval	
# 	
# def rupture_Duration(sumplot,samp_rate):
# 	### Duration method 1: Find first and last X(maxSTF) (Courboulex et al. 2016)
# 	cutOffSTF = 0.05*max(sumplot)
# 	#### Get indexes
# 	index_array = np.where(sumplot>cutOffSTF)
# 	delIdx = index_array[0][-1] - index_array[0][0]
# 	return delIdx*samp_rate,[index_array[0][0],index_array[0][-1]]
	
def rupture_Duration(sumplot,samp_rate,percent):
	### Duration method 1: Find first and last X(maxSTF) (Courboulex et al. 2016)
	cutOffSTF = percent*max(sumplot)
	#### Get indexes
	index_array = np.where(sumplot>cutOffSTF)
	delIdx = index_array[0][-1] #- index_array[0][0]
# 	return delIdx*samp_rate,[index_array[0][0],index_array[0][-1]]
	return delIdx*samp_rate,index_array[0][-1]
	
def rupture_Duration_StartUnk(sumplot,samp_rate,percent):
	### Duration method 1: Find first and last X(maxSTF) (Courboulex et al. 2016)
	cutOffSTF = percent*max(sumplot)
	#### Get indexes
	index_array = np.where(sumplot>cutOffSTF)
	delIdx = index_array[0][-1] - index_array[0][0]
	return delIdx*samp_rate,[index_array[0][0],index_array[0][-1]]
		

def stressCircSTF(T,Mo,k,beta,c):
	##### T: STF duration; Mo = Moment (N-m); k=fc to radius const.; beta= shear wave; c: fc = c/T
	s = (7./16.)*Mo*((c/(k*beta*T))**3)/(1e6) ### Dividing by 1e6 returns it in MPa
	return s
	
def stressCircfc(fc,Mo,k,beta):
	##### T: STF duration; Mo = Moment (N-m); k=fc to radius const.; beta= shear wave; c: fc = c/T
	s = (7./16.)*Mo*((fc/(k*beta))**3)/(1e6) ### Dividing by 1e6 returns it in MPa
	return s
	
def stress2fc(s,Mo,k,beta):
	fc = (((1e6)*s*(16./7.)/Mo)**(1./3.))*k*beta
	return fc
# 
# def stress2Moment(fc,k,beta,sigma):
# 	moment =  (16./7.)*sigma*(((k*beta)/fc)**3)
# 	return moment

def resampSpec(freq,amplitude_FT,log_samp):
	freq_log = np.log10(freq)
	freq_log_samp = np.arange(freq_log[0],freq_log[-1],log_samp)
	freq_samp = 10**freq_log_samp
	amp_samp = np.interp(freq_samp,freq,amplitude_FT)
	return freq_samp,amp_samp
	
def bruneModInv(coeffs,y,f):
	return np.log10(bruneMod(f,coeffs))- np.log10(y)
	
def bruneMod(f,coeffs):
	return coeffs[0]/(1+(f/coeffs[1])**2)
	
# def bruneModInvMoFixed(coeffs,y,f,Mo):
# 	return np.sum(np.square(np.log10(y) - np.log10(bruneModMoFixed(f,coeffs,Mo))))
	
def bruneModInvMoFixed(coeffs,y,f,Mo):
	return  np.log10(bruneModMoFixed(f,coeffs,Mo)) - np.log10(y)
	
def bruneModInvMoFixedslopeFree(coeffs,y,f,Mo):
	return  np.log10(bruneModMoFixedfslopeFree(f,coeffs,Mo)) - np.log10(y)
	
def bruneModInv_AllFree(coeffs,y,f):
	return  np.log10(bruneModAllFree(f,coeffs)) - np.log10(y)
	
def bruneModMoFixed(f,coeffs,Mo):
	return Mo/(1+(f/coeffs)**2) # Removed [0] after coeffs so may not work now. Double check
	
def bruneModMoFixedfslopeFree(f,coeffs,Mo):
	return Mo/(1+(f/coeffs[0])**coeffs[1])
	
	
def bruneModAllFree(f,coeffs):
	return coeffs[0]/(1+(f/coeffs[1])**coeffs[2])
	

def ratio_bruneModInvMoFixed(coeffs,y,f,Mo_main,Mo_EGF):
	return  np.log10(ratio_bruneModMoFixed(f,coeffs,Mo_main,Mo_EGF)) - np.log10(y)
	
def ratio_bruneModMoFixed(f,coeffs,Mo_main,Mo_EGF): # For ratio tests
	return bruneModMoFixed(f,coeffs[0],Mo_main)/bruneModMoFixed(f,coeffs[1],Mo_EGF)
	
def ratio_bruneModInvMoFixedslopeFree(coeffs,y,f,Mo_main,Mo_EGF):
	return  np.log10(ratio_bruneModMoFixedslopeFree(f,coeffs,Mo_main,Mo_EGF)) - np.log10(y)
	
def ratio_bruneModMoFixedslopeFree(f,coeffs,Mo_main,Mo_EGF): # For ratio tests
	return bruneModMoFixedfslopeFree(f,coeffs[0:2],Mo_main)/bruneModMoFixedfslopeFree(f,coeffs[2:],Mo_EGF)
	

	
# def REEF_f(ro,alpha,beta,STF,samp,T,Mo):
# # 	E_R = (1/(15*np.pi*ro*(alpha**5))+1/(10*np.pi*ro*(beta**5)))*integrate.simpson(np.gradient(STF,samp)**2,dx=samp) # calculate radiated energy
# 	E_R = (1/(10*np.pi*ro*(beta**5)))*integrate.simpson(np.gradient(STF,samp)**2,dx=samp) # calculate radiated energy
# 	E_Rmin = (6/(5*np.pi*ro*(beta**5)))*(Mo**2)/(T**3)
# 
# 	REEF = E_R/E_Rmin
# 	return REEF

def rough_fun(baseSTF,STF,samp):
	R_stf = integrate.simpson(np.gradient(STF,samp)**2,dx=samp)
	R_base = integrate.simpson(np.gradient(baseSTF,samp)**2,dx=samp)
	rough_fun = R_stf/R_base
	return rough_fun
	
	
### Radiated energy
def radE_STF(Mo,STF,samp,ro,alpha,beta):
	STF_der = np.gradient(STF,samp)**2
	radE = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5))))*integrate.simpson(STF_der,dx=samp) # Not normalized STF
# 	radE = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5))))*(Mo**2)*integrate.simpson(STF_der,dx=samp) # Normalized STF

	return radE


### Radiated energy New
def radE_STF_freq(disp,freq,ro,alpha,beta):
	vel =disp*freq # Convert displacement to velocity
	radE_freq = 8*np.pi*(1/(15*ro*(alpha**5))+(1/(10*ro*(beta**5))))*integrate.simpson(vel**2,freq) # Not Normalized STF
	return radE_freq	
	
	
def radE_STF_freq_cum(disp,freq,ro,alpha,beta):
	vel =disp*freq # Convert displacement to velocity
	radE_freq_cum = 8*np.pi*(1/(15*ro*(alpha**5))+(1/(10*ro*(beta**5))))*integrate.cumulative_trapezoid(vel**2,freq,initial=0) # Not Normalized STF
	return radE_freq_cum	
	


### Radiated energy OLD
# def radE_STF_freq(Mo,STF,timeVal,samp,ro,alpha,beta):
# 	STF_der = np.gradient(STF,samp)
# 	freq,amplitude_FT,FT_momentRate = sig_process(timeVal,STF_der,1/samp)
# 	radE_freq = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5))))*integrate.simpson(amplitude_FT**2,freq) # Not Normalized STF
# # 	radE_freq = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5))))*(Mo**2)*integrate.simpson(amplitude_FT**2,freq) # Normalized STF
# 	return radE_freq
	
def parab_stf(Mo,T,times):
	Mo_t = (6*Mo/T**3)*times*(T-times)
	return Mo_t

# def rough_amp_fun(baseSTF_amp,STF_amp,samp,T):
# 	R_stf = integrate.simpson(np.gradient(STF,samp)**2,dx=samp)
# 	R_base = integrate.simpson(np.gradient(baseSTF,samp)**2,dx=samp)
# 	rough_fun = R_stf/R_base
# 	return rough_fun
	
# def disp2mom_rate(disp_t,ro,beta):
# 	mom_rate = 4*np.pi*ro*disp_t*(beta**3)
# 	return mom_rate
# 
# def mom_rate2mom(mom_rate,dt):
# 	mom = np.trapz(mom_rate,dx=dt)
# 	return mom

def gen_noise(signal,SNR):
	### Function to calculate noise based on signal to noise ratio
	### See https://stats.stackexchange.com/questions/548619/how-to-add-and-vary-gaussian-noise-to-input-data for details
	Psig = np.mean(signal**2) # power of signal
	sigma_noise = np.sqrt(Psig/SNR) # Sigma of noise
	noise_vals = rng.normal(0,sigma_noise,len(signal)) # Array of noise random noise
	return noise_vals
	

### MSE
def NMSE_fun(res):
	NMSE = ((np.mean(res**2))**.5)
	return NMSE

### KS_Test
def KS_fun(data,model,samp_rate):
	cum_data = np.cumsum(data)*samp_rate
	cum_model = np.cumsum(model)*samp_rate
	KS = np.max(np.abs(cum_data-cum_model))
	return KS
	
### CM_Test
def CM_fun(data,model,samp_rate):
	cum_data = np.cumsum(data)*samp_rate
	cum_model = np.cumsum(model)*samp_rate
	CM = (np.sum((cum_data-cum_model)**2)/len(cum_model))**.5
	return CM
	
### Cross correlation max:
def xcorrMaxNorm_fun(data,model):
	xcorNorm = sig.correlate(model/np.linalg.norm(model), data/np.linalg.norm(data), mode='full', method='auto')
	xcorrMaxNorm = max(xcorNorm)
	return xcorrMaxNorm
	
### correlation 
def cor_fun(data,model):
	corr,pval = stats.pearsonr(model,data)
	return corr
	
	
#### Peak-to-peak
def p2p(res):
	# based on Uchide-Imanishi 2016 (ratio of max/min)
	return np.abs(max(res)-min(res))

### Normalize function	
def sig_normalize(STF,t):
	norm_STF = STF/integrate.simpson(STF,t)
	return norm_STF
	

	
### Trim STF
def trim_STF(STF,t,endpoints,samp_rate):
	T_length = (endpoints[1]- endpoints[0])*10 # Calculate length X10 (in endpoints)
	New_start = int(endpoints[0] - T_length/2)
	if New_start < 0: # If beyond bounds
		New_start = 0
	New_end = int(endpoints[1] + T_length/2)
	if New_end > len(STF):
		New_end = len(STF)-2
	STF_trim = STF[New_start:New_end+1]
	t_trim = np.arange(0,len(STF_trim))*samp_rate
	return t_trim,STF_trim


##### Energy calculations	
def Brune_Vel(Mo,f,fc): # EQ 1
	return Mo*f/(1+(f/fc))**2

def Brune_E(Mo,fc,ro,alpha,beta): # EQ 2
	return (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5))))*(1/4)*np.pi*(Mo**2)*(fc**3)
	
def F_fun(fm,fc):  # Eq. 3 It's wrong in the paper. Changed here to be within inner parenthesis
	F = (-fm/fc)/(1+(fm/fc)**2)+np.arctan2(fm,fc)
	return F
	
def Brune_E_Obs(Mo,fm,fc,ro,alpha,beta): #EQ 4
	F = F_fun(fm,fc)
	return (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5))))*(1/2)*(Mo**2)*(fc**3)*F
	
def mis_energy(fc,k,beta,fm): # EQ 5
	F = F_fun(fm,fc)
	R = (2/np.pi)*F # EQ 5
	return R
	
def var_energy(Mo,fc,fm,n,ro,alpha,beta):
	energy = (1/(15*np.pi*ro*(alpha**5))+(1/(10*np.pi*ro*(beta**5))))*(var_energy_int(Mo,fc,n,fm) - var_energy_int(Mo,fc,n,0))
	return energy
	
def var_energy_int(Mo,fc,n,f):
	integral = (Mo**2)*(f**3)*((n-3)*spec.hyp2f1(1,3/n,(n+3)/n,-(f/fc)**n)+3/(((f/fc)**n)+1))/(3*n)
	return integral
# 	
# def missing_e(Mo,fc,fm,n,ro,alpha,beta):
# 	e_meas = var_energy(Mo,fc,fm,n,ro,alpha,beta)/var_energy(Mo,fc,1e11,n,ro,alpha,beta) # Percent of energy measured
# 	return e_meas
# 	
### Based on mathematica
def missing_e(Mo,fc,fm,n): #,ro,alpha,beta):
	total_e = (-3+n)*np.pi*(Mo**2)*((fc**-n)**-(3/n))*(1/np.sin(3*np.pi/n))/(n**2)
	est_e = (Mo**2)*(fm**3)*(3/(1+(fm/fc)**n)+(-3+n)*spec.hyp2f1(1,3/n,(3+n)/n,-(fm/fc)**n))/(3*n)
	return est_e/total_e


##### PGA vs Mw vs stress drop(Boore 1983 eq. A10a)
def ref_pgaMw(M,S,refM,refS):
	log_aref = 0.31*(refM) + 0.8*np.log10(refS)
	log_a = 0.31*(M) + 0.8*np.log10(S)
	return 10**(log_a)/10**(log_aref)
	
	
##### Perform regression
def stress_complex_reg(x_in,y_in,x_pred):
	x_in = sm.add_constant(x_in)
	model = sm.OLS(y_in, x_in)
	results = model.fit()
	### Predict for values
	x_pred = sm.add_constant(x_pred)
	y_pred = model.predict(results.params,exog=x_pred)
	return results,y_pred
	

##### Sato and Hirasawa	
def sh_1973(a,v,st,theta,c,t,tshift):
	# from Sato and Hirasawa (1973) Udias equation. 7.67
	# a: fault radius
	# v: rupture velocity
	# st: static stress drop
	# theta: azimuthal direction
	# c: wave speed for p or s-wave
	# t: time after rupture commences
	# tshift: time shift
	t1 = (a/v)*(1-(v/c)*np.sin(theta))# Time to first stopping phase
	t2 = (a/v)*(1+(v/c)*np.sin(theta))# time to second stopping phase
	disp_1 = (24*st/7)*(a**2)*v*(2/(1-((v**2)/(c**2)*np.sin(theta)**2))**2)*((v**2)*(t[t<t1]**2)/(a**2))
	disp_2 = (24*st/7)*(a**2)*v*1/(2*(v/c)*np.sin(theta))*(1-(v**2)*(t[(t>t1)&(t<t2)]**2)/((a**2)*(1+v/c*np.sin(theta))**2))
	disp_temp = np.append(disp_1,disp_2)
	disp = np.append(disp_temp,np.zeros(len(t)-len(disp_temp)))
	if tshift > 0: # Check if start time shifted
		### Shift start time if necessary
		num_samp = int(tshift/(t[1]-t[0]))# Number of zeros to append
		disp_out = np.append(np.zeros(num_samp),disp[:-num_samp])
	else:
		disp_out = disp
# 	print(tshift)
# 	print(len(t))
# 	print(len(disp_out))
	return disp_out


##### Sato Radiii
def sato_radii(moment_sub,stress_sub):
	a =(moment_sub*(7/16)/stress_sub)**(1/3)
	return a


#### Theoretical Energy
def Hirano_Yagi_Energy(Mo,fmax,fc,n,g):
	### Hirano and Yagi 2017 R using eq 20
	F = (1+(fmax/fc)**(-g*n))**-1
	return (1/(g*n))*(Mo**2)*(fc**3)*spec.betainc(3/(g*n), 2/g-3/(g*n), F)*spec.beta(3/(g*n), 2/g-3/(g*n))
	
#### Attenuation operator
def atten_op_stein(x,c0,Q,freq_vals): # Stein and Wysession Es. 47-51 ch.3.7
	c_temp = c0*(1+np.log(freq_vals[1:]/freq_vals[1])/(np.pi*Q)) # Eq 51
	c = np.concatenate(([c_temp[0]],c_temp)) # Add in velocity for zeroth frequency
	delta_func = np.exp(-1j*2*np.pi*freq_vals*x/c) # Eq 47
	atten_func = np.exp(-2*np.pi*freq_vals*x/(2*Q*c)) # Eq 48
	atten_delta = atten_func*delta_func # Eq 49 Attenuator in fourier domain
	return atten_func,atten_delta,c
	
######## EGF ratio Radiated Energy
def EGF_energy_ratio(M_Mo,M_fc,M_n,E_Mo,E_fc,E_n,sig,freq,ro,alpha,beta):
	re_sig = radE_STF_freq(sig,freq,ro,alpha,beta)
	model = ratio_bruneModMoFixedslopeFree(freq,[M_fc,M_n,E_fc,E_n],M_Mo,E_Mo)
	re_model = radE_STF_freq(model,freq,ro,alpha,beta)
	return re_sig/re_model

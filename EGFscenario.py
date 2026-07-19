#### This script explores the impact of relative complexity on EGF estimates
import stress_func as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


###### Generate complex earthquakes

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
log_samp = 0.025 # Log sampling for fitting spectral model

def genSynth(numPulse,totalMw,pulseMomentpercent,pulseStressDrop,pulseTime):
    mo_main = sf.mw2mo(totalMw)
    moment_sub = mo_main*pulseMomentpercent
    ### Step 5: Calculate the radii of the sub-events
    radii_sub = sf.brune_radii(moment_sub, pulseStressDrop)
    area_sub = np.pi * np.square(radii_sub)
    ### Step 6: Calculate the corner frequencies of the sub-events
    fc_sub = sf.brune_fc(beta, radii_sub, k)
    #### Step 9: Creat the brune pulses
    pulse_sub = sf.brune_time_moment(np.atleast_2d(moment_sub).T, np.atleast_2d(fc_sub).T,
                                     np.atleast_2d(pulseTime).T + np.mean(t), np.tile(t, (int(numPulse),
                                                                                          1)))  # +50 on the start_sub ensures that it is shifted properly away from 0. Need to change shape for calculation
    pulse_sub[pulse_sub < 0] = 0  # Remove negative values

    #### Step 10: Sum the pulses for the final pulse
    pulse_sum = np.sum(pulse_sub, axis=0)
    ### Weighted Averages of stress drops
    sigma_Mo = np.sum((moment_sub / np.sum(moment_sub)) * pulseStressDrop)  # Moment weight average
    sigma_area = np.sum((area_sub / np.sum(area_sub)) * pulseStressDrop)  # Area weighted average
    return mo_main,moment_sub,radii_sub,area_sub,fc_sub,pulse_sub,pulse_sum,sigma_Mo,sigma_area


def brune_Gen(mw_tot,mag_dif,direction,simp_stress,freq):
    simp_mo = sf.mw2mo(mw_tot + direction*mag_dif)
    simp_mw = sf.mo2mw(simp_mo)
    simp_fc = sf.stress2fc(simp_stress, simp_mo, k, beta)
    sim_brune = sf.bruneMod(freq, [simp_mo, simp_fc])
    return sim_brune,simp_fc,simp_mo,simp_mw

#### Calculate spectral ratios
def spec_ratio(main,egf,main_mo,egf_mo,freq,log_samp):
    ratio = main/egf
    #### Resample in log space
    freq_smooth, ratio_smooth = sf.resampSpec(freq,ratio, log_samp)
    #### Fiting scenario 1: Assume known moment ratio
    initialGuess = [sf.brune_fc(beta, sf.brune_radii(main_mo, 1), k),
                    sf.brune_fc(beta, sf.brune_radii(egf_mo, 1),
                    k)]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
    outvalMoFixed = optimize.least_squares(sf.ratio_bruneModInvMoFixed, initialGuess, bounds=(.00001, np.inf),
                                 args=(ratio_smooth, freq_smooth, main_mo,
                                       egf_mo))
    est_fc_MoFixed_Main = outvalMoFixed.x[0]
    est_fc_MoFixed_EGF = outvalMoFixed.x[1]
    sigma_fc_est_Mofix_Main = sf.stressCircfc(est_fc_MoFixed_Main, main_mo, k, beta) # main
    sigma_fc_est_Mofix_EGF = sf.stressCircfc(est_fc_MoFixed_Main, egf_mo, k, beta) # egf

    #### #### #### #### #### #### moment ratio unknown and fixed decay of 2
    initialGuess = [sf.brune_fc(beta, sf.brune_radii(main_mo, 1), k),
                    sf.brune_fc(beta, sf.brune_radii(egf_mo, 1),k), 1]  ## Provide initial guess assume Stress Drop = 1 MPa for initial guess
    outvalMoFree = optimize.least_squares(sf.ratio_bruneModInvMoFree, initialGuess, bounds=(.00001, np.inf),
                                          args=(ratio_smooth, freq_smooth))
    est_fc_MoFree_Main = outvalMoFree.x[0]
    est_fc_MoFree_EGF = outvalMoFree.x[1]
    MoFree_ratio= outvalMoFree.x[2]
    sigma_fc_est_MoFree_Main = sf.stressCircfc(est_fc_MoFree_Main, main_mo, k,
                                                         beta)  # main
    sigma_fc_est_MoFree_EGF = sf.stressCircfc(est_fc_MoFree_EGF, egf_mo, k,
                                                        beta)  # egf
    return [est_fc_MoFixed_Main,est_fc_MoFixed_EGF,sigma_fc_est_Mofix_Main,sigma_fc_est_Mofix_EGF,est_fc_MoFree_Main,est_fc_MoFree_EGF,MoFree_ratio,sigma_fc_est_MoFree_Main,sigma_fc_est_MoFree_EGF]



#### hz band
HZ_Bands = [.01,5]

#### Generate Pulse 1
numPulse = 3
totalMw_1 = 6
pulseMomentpercent = np.array([.6,.1,.3])
pulseStressDrop = np.array([10,5,5])
pulseTime = np.array([0,2,1])
mo_main_1,moment_sub_1,radii_sub_1,area_sub_1,fc_sub_1,pulse_sub_1,pulse_sum_1,sigma_Mo_1,sigma_area_1 = genSynth(numPulse,totalMw_1,pulseMomentpercent,pulseStressDrop,pulseTime)
freq_1,amplitude_1,FT_complex = sf.sig_process(t,pulse_sum_1,[HZ_Bands[0],HZ_Bands[1]])


#### Generate Pulse 2
numPulse = 2
totalMw_2 = 5
pulseMomentpercent = np.array([.9,.1])
pulseStressDrop = np.array([10,10])
pulseTime = np.array([0,.5])
mo_main_2,moment_sub_2,radii_sub_2,area_sub_2,fc_sub_2,pulse_sub_2,pulse_sum_2,sigma_Mo_2,sigma_area_2 = genSynth(numPulse,totalMw_2,pulseMomentpercent,pulseStressDrop,pulseTime)
freq_2,amplitude_2,FT_complex = sf.sig_process(t,pulse_sum_2,[HZ_Bands[0],HZ_Bands[1]])


#### Generate Pulse 3
numPulse = 3
totalMw_3 = 7
pulseMomentpercent = np.array([.7,.3,.1])
pulseStressDrop = np.array([10,10,1])
pulseTime = np.array([0,1,1.5])
mo_main_3,moment_sub_3,radii_sub_3,area_sub_3,fc_sub_3,pulse_sub_3,pulse_sum_3,sigma_Mo_3,sigma_area_3 = genSynth(numPulse,totalMw_3,pulseMomentpercent,pulseStressDrop,pulseTime)
freq_3,amplitude_3,FT_complex = sf.sig_process(t,pulse_sum_3,[HZ_Bands[0],HZ_Bands[1]])

##### generate simple pulses
mag_dif = 1
simp_stress_small = 1
simp_stress_large = simp_stress_small
sim_brune_small,simp_fc_small,simp_mo_small,simp_mw_small = brune_Gen(totalMw_1,mag_dif,-1,simp_stress_small,freq_1)
sim_brune_large,simp_fc_large,simp_mo_large,simp_mw_large = brune_Gen(totalMw_1,mag_dif,1,simp_stress_large,freq_1)


### EGF spect ratio
ratio_1 = spec_ratio(sim_brune_large,amplitude_1,simp_mo_large,mo_main_1,freq_1,log_samp)
ratio_2 = spec_ratio(amplitude_1,sim_brune_small,mo_main_1,simp_mo_small,freq_1,log_samp)
ratio_3 = spec_ratio(amplitude_1,amplitude_2,mo_main_1,mo_main_2,freq_1,log_samp)
ratio_4 = spec_ratio(amplitude_3,amplitude_1,mo_main_3,mo_main_1,freq_1,log_samp)

# [est_fc_MoFixed_Main,est_fc_MoFixed_EGF,sigma_fc_est_Mofix_Main,sigma_fc_est_Mofix_EGF,est_fc_MoFree_Main,est_fc_MoFree_EGF,MoFree_ratio,sigma_fc_est_MoFree_Main,sigma_fc_est_MoFree_EGF]

bestFit_1 = sf.ratio_bruneModMoFixed(freq_1,[ratio_1[0],ratio_1[1]],simp_mo_large,mo_main_1)
bestFit_2 = sf.ratio_bruneModMoFixed(freq_1,[ratio_2[0],ratio_2[1]],mo_main_1,simp_mo_small)
bestFit_3 = sf.ratio_bruneModMoFixed(freq_1,[ratio_3[0],ratio_3[1]],mo_main_1,mo_main_2)
bestFit_4 = sf.ratio_bruneModMoFixed(freq_1,[ratio_4[0],ratio_4[1]],mo_main_3,mo_main_1)

bestFit_free_1 = sf.ratio_bruneModMoFree(freq_1,[ratio_1[4],ratio_1[5],ratio_1[6]])
bestFit_free_2 = sf.ratio_bruneModMoFree(freq_1,[ratio_2[4],ratio_2[5],ratio_2[6]])
bestFit_free_3 = sf.ratio_bruneModMoFree(freq_1,[ratio_3[4],ratio_3[5],ratio_3[6]])
bestFit_free_4 = sf.ratio_bruneModMoFree(freq_1,[ratio_4[4],ratio_4[5],ratio_4[6]])
#### Create Figures
fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(16,16))
ax = ax.flatten()

#### STF Time domain
ax[0].plot(t,pulse_sum_1,color='blue')
ax[0].set_title('MW: {:.1f}'.format(totalMw_1))
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Moment')
ax[0].set_xlim([max(t)/2-5,max(t)/2 +50])

ax[1].plot(t,pulse_sum_2,color='green')
ax[1].set_title('MW: {:.1f}'.format(totalMw_2))
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Moment')
ax[1].set_xlim([max(t)/2-5,max(t)/2 +50])


ax[2].plot(t,pulse_sum_3,color='orange')
ax[2].set_title('MW: {:.1f}'.format(totalMw_3))
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Moment')
ax[2].set_xlim([max(t)/2-5,max(t)/2 +50])
#### STF Freq domain
ax[3].loglog(freq_1,amplitude_1,color='blue',lw=3,label='MW: {:.1f}'.format(totalMw_1))
ax[3].loglog(freq_1,sim_brune_small,color='red',lw=3,label='MW: {:.1f}, Sig: {:.1f}MPa'.format(simp_mw_small,simp_stress_small))
ax[3].loglog(freq_1,sim_brune_large,color='purple',lw=3,label='MW: {:.1f}, Sig: {:.1f}MPa'.format(simp_mw_large,simp_stress_large))
ax[3].legend()

ax[4].loglog(freq_1,amplitude_1,color='blue',lw=3,label='MW: {:.1f}'.format(totalMw_1))
ax[4].loglog(freq_2,amplitude_2,color='green',lw=3,label='MW: {:.1f}'.format(totalMw_2))
ax[4].loglog(freq_1,sim_brune_large,color='purple',lw=3,label='MW: {:.1f}, Sig: {:.1f}MPa'.format(simp_mw_large,simp_stress_large))
ax[4].legend()

ax[5].loglog(freq_1,amplitude_1,color='blue',lw=3,label='MW: {:.1f}'.format(totalMw_1))
ax[5].loglog(freq_2,amplitude_2,color='green',lw=3,label='MW: {:.1f}'.format(totalMw_2))
ax[5].loglog(freq_1,amplitude_3,color='orange',lw=3,label='MW: {:.1f}'.format(totalMw_3))
ax[5].legend()


#### STF Ratios
ax[6].loglog(freq_1,amplitude_1/sim_brune_small,color='red',lw=3,label='MW: {:.1f}/MW: {:.1f}, Sig: {:.1f}MPa MAIN'.format(totalMw_1,simp_mw_small,simp_stress_small))
ax[6].loglog(freq_1,sim_brune_large/amplitude_1,color='purple',lw=3,label='MW: {:.1f}, Sig: {:.1f}MPa/MW: {:.1f} EGF'.format(simp_mw_large,simp_stress_large,totalMw_1))

ax[6].loglog(freq_1,bestFit_1,color='purple',ls=':')
ax[6].loglog(freq_1,bestFit_free_1,color='purple',ls='--')
ax[6].loglog(freq_1,bestFit_2,color='red',ls=':')
ax[6].loglog(freq_1,bestFit_free_2,color='red',ls='--')
ax[6].axvline(ratio_1[1],color='purple',ls=':')
ax[6].axvline(ratio_1[5],color='purple',ls='--')
ax[6].axvline(ratio_2[0],color='red',ls=':')
ax[6].axvline(ratio_2[4],color='red',ls='--')
ax[6].legend()


# est_fc_MoFixed_Main,est_fc_MoFixed_EGF,sigma_fc_est_Mofix_Main,sigma_fc_est_Mofix_EGF,est_fc_MoFree_Main,est_fc_MoFree_EGF,MoFree_ratio,sigma_fc_est_MoFree_Main,sigma_fc_est_MoFree_EGF]

ax[7].loglog(freq_1,amplitude_1/amplitude_2,color='green',lw=3,label='MW: {:.1f}/MW: {:.1f} MAIN'.format(totalMw_1,totalMw_2))
ax[7].loglog(freq_1,sim_brune_large/amplitude_1,color='purple',lw=3,label='MW: {:.1f}, Sig: {:.1f}MPa/MW: {:.1f} EGF'.format(simp_mw_large,simp_stress_large,totalMw_1))
ax[7].loglog(freq_1,bestFit_1,color='purple',ls=':')
ax[7].loglog(freq_1,bestFit_free_1,color='purple',ls='--')
ax[7].loglog(freq_1,bestFit_3,color='green',ls=':')
ax[7].loglog(freq_1,bestFit_free_3,color='green',ls='--')

ax[7].axvline(ratio_1[1],color='purple',ls=':')
ax[7].axvline(ratio_1[5],color='purple',ls='--')
ax[7].axvline(ratio_3[0],color='green',ls=':')
ax[7].axvline(ratio_3[4],color='green',ls='--')
ax[7].legend()


ax[8].loglog(freq_1,amplitude_1/amplitude_2,color='green',lw=3,label='MW: {:.1f}/MW: {:.1f} MAIN'.format(totalMw_1,totalMw_2))
ax[8].loglog(freq_1,amplitude_3/amplitude_1,color='orange',lw=3,label='MW: {:.1f}/MW: {:.1f} EGF'.format(totalMw_3,totalMw_1))
ax[8].loglog(freq_1,bestFit_4,color='orange',ls=':')
ax[8].loglog(freq_1,bestFit_free_4,color='orange',ls='--')
ax[8].loglog(freq_1,bestFit_3,color='green',ls=':')
ax[8].loglog(freq_1,bestFit_free_3,color='green',ls='--')

ax[8].axvline(ratio_4[1],color='orange',ls=':')
ax[8].axvline(ratio_4[5],color='orange',ls='--')
ax[8].axvline(ratio_3[0],color='green',ls=':')
ax[8].axvline(ratio_3[4],color='green',ls='--')
ax[8].legend()



fig.tight_layout()
fig.savefig('/Users/jamesneely/Documents/NSF/StressDrop_Bands/EGFAnalysis/EGFPlot_'+'{:.3f}Hz_{:.3f}HZ_'.format(HZ_Bands[0],HZ_Bands[1])+'.png',dpi=500)



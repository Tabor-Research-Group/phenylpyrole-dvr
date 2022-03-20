import matplotlib.pyplot as plt
import numpy as np
import random

import scipy

from scipy import interpolate

import pandas as pd

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.interpolate import interp1d

global AU_TO_WAVENUMBER 

AU_TO_WAVENUMBER = 219474.63

# Writes file to run the periodic DVR calculation
# 


def setup_dvr(num_states=0,num_grid_points=0,mass=0.0,
                         potential_energy_file_name=None,
                        output_file_name=None):
    f = open("input_period.dat",'w')
    f.write("Blank\n")  # First line is blank
    f.write(str(num_states)+"\n") # Second line is how many states
     # These two lines below are for specifying an interval
     # Currently the interval is taken for the full interval 
    # but written so it can be adapted 
    f.write("0.0 \n")   
    f.write("6.28 \n")
    f.write(str(num_grid_points)+"\n")
    f.write(str(mass)+"\n")
    f.write(output_file_name+"\n")
    f.close()
    return 

# Code assumes degrees and au for potential 
# Num grid points is actually 2N+1 (self-consistent with Colbert and Miller definition)
def fit_linear_potential_assume_periodic_write_to_file(raw_potential_filename=None,num_grid_points=0,
                                output_potential_filename=None,shift_origin=0,scale_factor=1.0):

    potential_angles = np.loadtxt(raw_potential_filename,usecols=0)
    potential_angles = potential_angles-shift_origin
    potential_energies = np.loadtxt(raw_potential_filename,usecols=1)
    # convert 
    potential_energies = potential_energies-min(potential_energies)
    potential_angles =  potential_angles*np.pi/180.0  

    #interpolate potential
    f2 = interp1d(potential_angles, potential_energies, kind='linear')
                        
    # Define pot for DVR
    angle_min = 0.0
    angle_max = 2.0*np.pi
    num_points = num_grid_points*2+1
    delta_angle = (angle_max-angle_min)/float(num_points)
    angles = np.arange(angle_min+delta_angle,angle_max+delta_angle,delta_angle)
                                
    pes_interpolate_values = f2(angles)
    f = open(output_potential_filename,'w')
    for i in range(0,angles.size):
        f.write(str(angles[i])+" "+str(pes_interpolate_values[i]*scale_factor)+'\n')
    f.close()
    
    
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 


    ax.set_xlabel('Angle (Radians)', fontsize = 12)
    ax.set_ylabel('Potential', fontsize = 12)
    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(0,max(pes_interpolate_values*AU_TO_WAVENUMBER)*1.1)
 
    ax.plot(angles, pes_interpolate_values*AU_TO_WAVENUMBER*scale_factor,label='Fitted/Scaled Potential', color='r',ls='-')
    ax.scatter((potential_angles+6.0*np.pi)%(2.0*np.pi), potential_energies*AU_TO_WAVENUMBER,label='Raw Values', color='k')


    ax.legend()
    
    return 


# Returns a dictionary that has the state, energy, and wavefunction
def read_dvr_output(filename_root=None,num_states=0):
    dvr_dict = dict()
    dvr_energies = np.loadtxt(filename_root+'_dvr_energies.out',usecols=0)
    for i in range(0,num_states):
        dvr_wavefunction_angles = np.loadtxt(filename_root+'_wavefunction_'+str(i+1)+'.dat',usecols=0)
        dvr_wavefunction_values = np.loadtxt(filename_root+'_wavefunction_'+str(i+1)+'.dat',usecols=1)
        temp_dict = dict()
        temp_dict['energy'] = dvr_energies[i]
        temp_dict['wavefunction_angles'] = dvr_wavefunction_angles
        temp_dict['wavefunction_values'] = dvr_wavefunction_values
        dvr_dict[filename_root+'_'+str(i)] = temp_dict
    
    return dvr_dict


def calculate_wavefunction_overlap(wavefunction_1_angles=None,
                                   wavefunction_1_values=None,
                                  wavefunction_2_angles=None,
                                   wavefunction_2_values=None):
    
    num_elements = wavefunction_1_angles.size

    overlap_integral = 0.0
    
    total_length = 0.0 
    
    # Check for self-consistency
    
    for i in range(0,num_elements-1):
        if abs(wavefunction_1_angles[i]-wavefunction_2_angles[i]) > 0.0001:
            
            print('Warning, there is a disagreement in the value of the angle!')
            print(str(wavefunction_1_angles[i]))
            print(str(wavefunction_2_angles[i]))
        
        # this should be a constant, but we'll do it this way anyway 
        length = wavefunction_1_angles[i+1]-wavefunction_1_angles[i]
        
        current_contribution = wavefunction_1_values[i]*wavefunction_2_values[i]*length
        
        overlap_integral = overlap_integral + current_contribution
    
        total_length = total_length + length

    return overlap_integral

def plot_wavefunction_squared(state_dict=None,state_num=0,prefix=None,
                      plotting_potential_name=None,
                     potential=None,mass=None):
    
    fig = plt.figure(figsize = (7.5,7.5))
    ax = fig.add_subplot(1,1,1) 


    ax.set_xlabel('Angle (Radians)', fontsize = 12)
    ax.set_ylabel('Potential', fontsize = 12)
    ax.set_xlim(0,2*np.pi)

    # read in potential 
    fitted_angle = np.loadtxt(plotting_potential_name,usecols=0)
    fitted_energy = np.loadtxt(plotting_potential_name,usecols=1)

    ax.plot(fitted_angle, fitted_energy*AU_TO_WAVENUMBER,label='potential', color='r',ls='-')

    print('Examining state '+prefix+str(state_num))
    
    # plot the energy for the state of interest
    
    state_information = state_dict[prefix+str(state_num)]
    energy = state_information['energy']
    plt.hlines(energy*AU_TO_WAVENUMBER,0,2.0*np.pi)

    # Now read in wavefunctions.
    
    ax2 = ax.twinx()

    state_information = state_dict[prefix+str(state_num)]
    
    
    
    angles = state_information['wavefunction_angles']
    wavefunction = state_information['wavefunction_values']
    ax2.plot(angles, wavefunction*wavefunction, '-' ,label='State '+str(state_num))
    ax.legend(loc='upper left')
    
    ax2.legend(loc='upper right')
    plt.savefig('./'+str(prefix)+'_state_pes_and_wavefunction_squared_mass_'+str(mass)+'state_'+str(state_num)+'.pdf')
    plt.show()
    
    return


# This function reads in ground and excited wavefunction data files
# and calculates the overlaps between the wavefunctions

def read_wavefunctions_calculate_overlaps_vib_shift(ground_state_file_name_root=None,
                                         excited_state_file_name_root=None,
                                         max_num_ground_state_wfs=0,
                                         max_num_excited_state_wfs=0,
                                         ground_state_max_energy=0,
                                         excited_state_max_energy=0,
                                                   vib_shift=0,
                                                   vib_scale=1.0):
    
# now read in wavefunctions and energies and calculate
    ground_state_dict = read_dvr_output(filename_root=ground_state_file_name_root,num_states=max_num_ground_state_wfs)
    excited_state_dict = read_dvr_output(filename_root=excited_state_file_name_root,num_states=max_num_excited_state_wfs)

# Normalize ground electronic state nuclear wavefunctions

    for state_key,state_information in ground_state_dict.items():
        normalization_constant = calculate_wavefunction_overlap(wavefunction_1_angles=state_information['wavefunction_angles'],
                                   wavefunction_1_values=state_information['wavefunction_values'],
                                  wavefunction_2_angles=state_information['wavefunction_angles'],
                                   wavefunction_2_values=state_information['wavefunction_values'])
    
        state_information['wavefunction_values'] = state_information['wavefunction_values']/np.sqrt(normalization_constant)
    
    # Normalize excited state

    for state_key,state_information in excited_state_dict.items():
        normalization_constant = calculate_wavefunction_overlap(wavefunction_1_angles=state_information['wavefunction_angles'],
                                   wavefunction_1_values=state_information['wavefunction_values'],
                                  wavefunction_2_angles=state_information['wavefunction_angles'],
                                   wavefunction_2_values=state_information['wavefunction_values'])
    
    
        state_information['wavefunction_values'] = state_information['wavefunction_values']/np.sqrt(normalization_constant)
    
    # Need to calculate pairs of overlaps. This is an N^2 process
    # But it can be pruned

    overlap_dicts =list() # This is a list of dicts. 

    # Find the minimum ground state energy 
    
    min_ground_energy = 9999999
    
    for ground_state_key,ground_state_information in ground_state_dict.items():
        if ground_state_information['energy'] < min_ground_energy:
            min_ground_energy = ground_state_information['energy']
    
    # Find the minimum excited state energy 
    
    min_excited_energy = 99999999
    
    for excited_state_key,excited_state_information in excited_state_dict.items():
        if excited_state_information['energy'] < min_excited_energy:
            min_excited_energy = excited_state_information['energy']
    
    for ground_state_key,ground_state_information in ground_state_dict.items():
        for excited_state_key,excited_state_information in excited_state_dict.items():
            current_dict = dict()
            # Check the energy 
            if ground_state_information['energy']*AU_TO_WAVENUMBER < ground_state_max_energy:
                if excited_state_information['energy']*AU_TO_WAVENUMBER < excited_state_max_energy:
                    relative_energy = excited_state_information['energy'] - min_excited_energy
                    # correct for being above the ground state
                    
                    relative_energy = relative_energy - (ground_state_information['energy']-min_ground_energy)

                    print('Calculating overlap for transition with energy '+str(relative_energy*AU_TO_WAVENUMBER))

                    
                    overlap = calculate_wavefunction_overlap(wavefunction_1_angles=ground_state_information['wavefunction_angles'],
                                   wavefunction_1_values=ground_state_information['wavefunction_values'],
                                  wavefunction_2_angles=excited_state_information['wavefunction_angles'],
                                   wavefunction_2_values=excited_state_information['wavefunction_values'])
                    
                    # now add to dictionary 
                    current_dict['energy'] = relative_energy
                    current_dict['ground_state_id'] = ground_state_key
                    current_dict['excited_state_id'] = excited_state_key
                    current_dict['overlap'] = abs(overlap)**2
                    
                    
                    
                    overlap_dicts.append(current_dict)
                    
                    # now add the vibrational shifted level 
                    
                    if excited_state_information['energy']*AU_TO_WAVENUMBER+vib_shift < excited_state_max_energy:
                        current_dict = dict()
                        current_dict['energy'] = relative_energy + vib_shift/AU_TO_WAVENUMBER
                        current_dict['ground_state_id'] = ground_state_key+'_vib_excited'
                        current_dict['excited_state_id'] = excited_state_key+'_vib_excited'
                        current_dict['overlap'] = abs(overlap)**2*vib_scale
                        overlap_dicts.append(current_dict)
                    
    return overlap_dicts



def plot_wavefunction(state_dict=None,state_num=0,prefix=None,
                      plotting_potential_name=None,
                     potential=None,mass=None):
    
    fig = plt.figure(figsize = (7.5,7.5))
    ax = fig.add_subplot(1,1,1) 


    ax.set_xlabel('Angle (Radians)', fontsize = 12)
    ax.set_ylabel('Potential', fontsize = 12)
    ax.set_xlim(0,2*np.pi)

    # read in potential 
    fitted_angle = np.loadtxt(plotting_potential_name,usecols=0)
    fitted_energy = np.loadtxt(plotting_potential_name,usecols=1)

    ax.plot(fitted_angle, fitted_energy*AU_TO_WAVENUMBER,label='potential', color='r',ls='-')

    # Plots all energy levels with the wavefunctions
    for state_key,state_information in state_dict.items():
        energy = state_information['energy']
        if energy*AU_TO_WAVENUMBER < 4000:
            plt.hlines(energy*AU_TO_WAVENUMBER,0,2.0*np.pi)

    # Now read in wavefunctions.

    ax2 = ax.twinx()

    state_information = state_dict[prefix+str(state_num)]
    
    print('Examining state '+prefix+str(state_num))
    
    angles = state_information['wavefunction_angles']
    wavefunction = state_information['wavefunction_values']
    ax2.plot(angles, wavefunction, '-',label='State '+str(state_num))
    ax.legend(loc='upper left')
    
    ax2.legend(loc='upper right')
    plt.savefig('./'+str(prefix)+'_state_pes_and_wavefunctions_mass_'+str(mass)+'state_'+str(state_num)+'.pdf')
    plt.show()
    
    return



def fit_potential_write_to_file(raw_potential_filename=None,num_grid_points=0,
                                output_potential_filename=None,scale_factor=1.0):

    potential_angles = np.loadtxt(raw_potential_filename,usecols=0)
    potential_energies = np.loadtxt(raw_potential_filename,usecols=1)
    # convert 
    potential_energies = potential_energies-min(potential_energies)
    potential_angles =  potential_angles*np.pi/180.0  

    #interpolate potential
    f2 = interp1d(potential_angles, potential_energies, kind='linear')
                              
    # Define pot for DVR
    angle_min = 0.0
    angle_max = 2.0*np.pi
    num_points = num_grid_points*2+1
    delta_angle = (angle_max-angle_min)/float(num_points)
    angles = np.arange(angle_min+delta_angle,angle_max+delta_angle,delta_angle)
                                
    pes_interpolate_values = f2(angles)
    f = open(output_potential_filename,'w')
    for i in range(0,angles.size):
        f.write(str(angles[i])+" "+str(pes_interpolate_values[i]*scale_factor)+'\n')
    f.close()

    return

def plot_potential_and_energy_levels(state_dict=None,prefix=None,
                      plotting_potential_name=None,
                     potential=None,mass=None,max_energy=0.0):
    
    fig = plt.figure(figsize = (7.5,7.5))
    ax = fig.add_subplot(1,1,1) 


    ax.set_xlabel('Angle (Radians)', fontsize = 20)
    ax.set_ylabel('Potential cm$^{-1}$', fontsize = 20)
  #  ax.set_title('\"Mass" set to '+str(mass), fontsize = 12)
    ax.set_xlim(0,2*np.pi)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # read in potential 
    fitted_angle = np.loadtxt(plotting_potential_name,usecols=0)
    fitted_energy = np.loadtxt(plotting_potential_name,usecols=1)

    ax.plot(fitted_angle, fitted_energy*AU_TO_WAVENUMBER,label='potential', color='r',ls='-')

    for state_key,state_information in state_dict.items():
        energy = state_information['energy']
        if energy*AU_TO_WAVENUMBER < max_energy:
            plt.hlines(energy*AU_TO_WAVENUMBER,0,2.0*np.pi)


    plt.savefig('./'+str(prefix)+'_pes_and_energy_levels.pdf',bbox_inches='tight')
    plt.show()


# Unused in final paper. 
def arbitrary_grid_linear_potential_write_to_file(raw_potential_filename=None,
                                                  num_grid_points=0,
                                output_potential_filename=None,scale_factor=1.0):

    potential_angles = np.loadtxt(raw_potential_filename,usecols=0)
    potential_energies = np.loadtxt(raw_potential_filename,usecols=1)
    # convert 
    potential_energies = potential_energies-min(potential_energies)
    potential_angles =  potential_angles*np.pi/180.0  

    #interpolate potential
                              
    # Define potential for DVR
    angle_min = 0.0
    angle_max = 2.0*np.pi
    num_points = num_grid_points*2+1
    delta_angle = (angle_max-angle_min)/float(num_points)
    angles = np.arange(angle_min+delta_angle,angle_max+delta_angle,delta_angle)
                                
    f = open(output_potential_filename,'w')
    for i in range(0,angles.size):
        # Find the two angles that the current angle is in between        
        angle_1_pos = -1
        angle_2_pos = -1
        
        for j in range(0,angles.size):
            if angle_1_pos == -1 and angles[i] > potential_angles[j]:
                angle_1_pos = j
            
        
        f.write(str(angles[i])+" "+str(pes_interpolate_values[i]*scale_factor)+'\n')
    f.close()

    return

# Code assumes degrees and au for potential 
# Num grid points is actually 2N+1
# This is for periodic conditions
# Potential must have the same first and last point!
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
def fit_potential_spline_assume_periodic_write_to_file(raw_potential_filename=None,num_grid_points=0,
                                output_potential_filename=None,shift_origin=0,scale_factor=1.0):

    potential_angles = np.loadtxt(raw_potential_filename,usecols=0)
    potential_angles = potential_angles-shift_origin
    potential_energies = np.loadtxt(raw_potential_filename,usecols=1)
    # convert 
    potential_energies = potential_energies-min(potential_energies)
    potential_angles =  potential_angles*np.pi/180.0  

    #interpolate potential
    fcubic = scipy.interpolate.CubicSpline(potential_angles, potential_energies, extrapolate='periodic',bc_type='periodic')
  #  fquadratic = scipy.interpolate.(potential_angles, potential_energies, extrapolate='periodic',bc_type='periodic')                    
    #f2 = scipy.interpolate.Spline(potential_angles, potential_energies, extrapolate='periodic',bc_type='periodic')
    # Define pot for DVR
    angle_min = 0.0
    angle_max = 2.0*np.pi
    num_points = num_grid_points*2+1
    delta_angle = (angle_max-angle_min)/float(num_points)
    angles = np.arange(angle_min+delta_angle,angle_max+delta_angle,delta_angle)
                                
    pes_interpolate_values_cubic = fcubic(angles)
    #f = open(output_potential_filename,'w')
    #for i in range(0,angles.size):
    #    f.write(str(angles[i])+" "+str(pes_interpolate_values[i]*scale_factor)+'\n')
    #f.close()
    
    
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 


    ax.set_xlabel('Angle (Radians)', fontsize = 12)
    ax.set_ylabel('Potential (cm$^{-1}$)', fontsize = 12)
    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(min(0,min(pes_interpolate_values_cubic*AU_TO_WAVENUMBER)),max(pes_interpolate_values_cubic*AU_TO_WAVENUMBER)*1.1)
 
    #ax.plot(angles, kidwell_gs_with_h2o_interpolate_values*AU_TO_WAVENUMBER,label='Kidwell GS Potential with Water', color='r',ls='-')
    ax.plot(angles, pes_interpolate_values_cubic*AU_TO_WAVENUMBER*scale_factor,label='Fitted Potential Cubic', color='r',ls='-')
    ax.scatter((potential_angles+6.0*np.pi)%(2.0*np.pi), potential_energies*AU_TO_WAVENUMBER,label='Raw Values', color='k')


    ax.legend()
    
#   Write to file
    f = open(output_potential_filename,'w')
    for i in range(0,angles.size):
        f.write(str(angles[i])+" "+str(pes_interpolate_values_cubic[i]*scale_factor)+'\n')
    f.close()

    return 





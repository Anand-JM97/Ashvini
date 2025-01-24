# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:47:28 2024

Main code with evolve galaxy function

@author: Anand Menon
"""

#PYTHON PACKAGES

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import units as cu
from astropy.cosmology import z_at_value
import astropy.units as u

from scipy.integrate import solve_ivp

#PACKAGES FROM ASHVINI

from io import mdot_h,m_h
import reionization as rei
import star_formation as sf
import supernovae_feedback as snw

H_0=cosmo.H0 #in km / (Mpc s)
H_0=H_0.to(u.Gyr**(-1)) #in 1/Gyr

A=0.030

omega_m=cosmo.Om0
omega_b=cosmo.Ob0
omega_L=cosmo.Ode0

pi=np.pi

plt.rcParams['figure.dpi'] = 300

#t=cosmo.age(z)

#z=z_at_value(cosmo.age,t)

def t(z):
    """
    Function to convert redshift to cosmic time.
    Args:
        z (float): Parameter representing redshift.

    Returns:
        Float: The comsic time value.
    """
    t_val=cosmo.age(z)
    return t_val

def z(t):
    """
    Function to convert cosmic time to redshift.
    Args:
        t (float): Parameter representing cosmic time.

    Returns:
        Float: The redshift value.
    """
    z_val=z_at_value(cosmo.age,t)
    return z_val

def H(z):
    """
    Hubble function.
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The value of the Hubble constant for the redshift value entered as the argument.
    """
    H=cosmo.H(z)
    H=H.to(u.Gyr**(-1)) #in 1/Gyr
    return H

def evolve_galaxies():
    """This will be the main function. We should have this call star formation, supernovae feedback, and such defined
    in the other files. Ideally this should be the only the function that is defined in this file.
    """

    return 1


#Initialization for cosmic time 

t0=0.129
tf=5
n=25000
h=(tf-t0)/n
cosmic_time=np.linspace(t0,tf,n)

#Metallicity parameters

z_igm=10**(-3)
y_z=0.06
zeta_w=1


def m_dot_cg_with_UV(z,m_h0_val):
    m_dot_cg_val=(omega_b/omega_m)*mdot_h(z,m_h0_val)*rei.epsilon_uv(z,m_h0_val)
    return m_dot_cg_val

def m_dot_cg_no_UV(z,m_h0_val):
    m_dot_cg_val=(omega_b/omega_m)*mdot_h(z,m_h0_val)
    return m_dot_cg_val


def diff_eqns_1(t,r,m_h0_val,e_ff,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg(z_val,m_h0_val)-(e_ff/sf.t_ff(z_val))*m_g
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
        
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_2(t,r,m_d_s_d,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg(z_val,m_h0_val)-(e_ff/sf.t_ff(z_val))*m_g-snw.eta(z_val,m_h0_val)*m_d_s_d
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
    
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_3(t,r,m_h0_val,e_ff,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg_2(z_val,m_h0_val)-(e_ff/sf.t_ff(z_val))*m_g
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
        
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_4(t,r,m_d_s_d,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg_2(z_val,m_h0_val)-(e_ff/sf.t_ff(z_val))*m_g-snw.eta(z_val,m_h0_val)*m_d_s_d
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
        
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_eq_1(t,r,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)

    f_m_g=m_dot_cg(z_val,m_h0_val)-(e_ff/sf.t_ff(z_val))*m_g-snw.eta(z_val,m_h0_val)*(e_ff/sf.t_ff(z_val))*m_g
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
            
    return(np.array([f_m_g,f_m_star]))    

def diff_eqns_eq_2(t,r,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)

    f_m_g=m_dot_cg_2(z_val,m_h0_val)-(e_ff/sf.t_ff(z_val))*m_g-snw.eta(z_val,m_h0_val)*(e_ff/sf.t_ff(z_val))*m_g
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
    
    return(np.array([f_m_g,f_m_star]))    

def diff_eqn_zgas_1(t,y,m_h0_val):  #diff_eqns_1
    z_val=z(t)
    
    f_m_z_gas=z_igm*m_dot_cg(z_val,m_h0_val)
    return f_m_z_gas

def diff_eqn_zgas_2(t,y,m_g,m_h0_val,e_ff,gamma_ff):    #diff_eqns_1
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))-((y/m_g)*(e_ff/sf.t_ff(z_val))*m_g)
    return f_m_z_gas

def diff_eqn_zgas_3(t,y,m_h0_val,m_d_s_d):      #diff_eqns_2
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))+(y_z*m_d_s_d)
    return f_m_z_gas

def diff_eqn_zgas_4(t,y,m_g,m_h0_val,m_d_s_d,e_ff,gamma_ff,epsilon_p):      #diff_eqns_2
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))-(y*(e_ff/sf.t_ff(z_val)))+(y_z*m_d_s_d)-(snw.eta(z_val,m_h0_val)*(y/m_g)*m_d_s_d)
    return f_m_z_gas

def diff_eqn_zgas_5(t,y,m_h0_val):       #diff_eqns_3
    z_val=z(t)

    f_m_z_gas=z_igm*m_dot_cg_2(z_val,m_h0_val)
    return f_m_z_gas

def diff_eqn_zgas_6(t,y,m_g,m_h0_val,e_ff,gamma_ff):      #diff_eqns_3
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))-((y/m_g)*(e_ff/sf.t_ff(z_val))*m_g)
    return f_m_z_gas

def diff_eqn_zgas_7(t,y,m_h0_val,m_d_s_d):       #diff_eqns_4
    z_val=z(t)

    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))+(y_z*m_d_s_d)
    return f_m_z_gas

def diff_eqn_zgas_8(t,y,m_g,m_h0_val,m_d_s_d,e_ff,gamma_ff,epsilon_p):       #diff_eqns_4
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))-(y*(e_ff/sf.t_ff(z_val)))+(y_z*m_d_s_d)-(snw.eta(z_val,m_h0_val)*(y/m_g)*m_d_s_d)
    return f_m_z_gas

def diff_eqn_eq_zgas_1(t,y,m_g,m_h0_val,e_ff,gamma_ff):       #diff_eqns_eq_1
    z_val=z(t)
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))+(y_z*f_m_star)
    return f_m_z_gas
    
def diff_eqn_eq_zgas_2(t,y,m_g,m_h0_val,e_ff,gamma_ff,epsilon_p):        #diff_eqns_eq_1
    z_val=z(t)
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))-(y*f_m_star/m_g)+(y_z*f_m_star)-(snw.eta(z_val,m_h0_val)*y*f_m_star/m_g)
    return f_m_z_gas

def diff_eqn_eq_zgas_3(t,y,m_g,m_h0_val,e_ff,gamma_ff):       #diff_eqns_eq_2
    z_val=z(t)
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))+(y_z*f_m_star)
    return f_m_z_gas
    
def diff_eqn_eq_zgas_4(t,y,m_g,m_h0_val,e_ff,gamma_ff,epsilon_p):        #diff_eqns_eq_2
    z_val=z(t)
    f_m_star=(e_ff/sf.t_ff(z_val))*m_g
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))-(y*f_m_star/m_g)+(y_z*f_m_star)-(snw.eta(z_val,m_h0_val)*y*f_m_star/m_g)
    return f_m_z_gas

def diff_eqn_zstar_1(t,y):
    f_m_z_star=0.0
    return f_m_z_star

def diff_eqn_zstar_2(t,y,m_z_g,m_h0_val,e_ff,gamma_ff):
    z_val=z(t)
    f_m_z_star=(m_z_g)*(e_ff/sf.t_ff(z_val))
    return f_m_z_star

#IVF SOLVER FUNCTIONS

#NO FEEDBACK


def diff_eqn_gas_1(t,y,m_d_cg): #diff_eqns_1
    z_val=z(t)
    
    f_m_gas=m_d_cg-(sf.e_ff/sf.t_ff(z_val))*y
    return f_m_gas

def diff_eqn_star_1(t,y,m_g):  #diff_eqns_1
    z_val=z(t)
    
    f_m_star=(sf.e_ff/sf.t_ff(z_val))*m_g
    return f_m_star


def diff_eqn_zgas_1(t,y,m_g,m_d_cg):    #diff_eqns_1
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_d_cg)-((y)*(sf.e_ff/sf.t_ff(z_val)))
    return f_m_z_gas

    
#DELAYED FEEDBACK


def diff_eqn_gas_2(t,y,m_d_cg,m_halo,m_d_s_d,z_star):       #diff_eqns_2
    z_val=z(t)
    
    f_m_gas=m_d_cg-(sf.e_ff/sf.t_ff(z_val))*y-snw.eta(z_val,m_halo,z_star)*m_d_s_d
    return f_m_gas

def diff_eqn_star_2(t,y,m_g):       #diff_eqns_2
    z_val=z(t)
    
    f_m_star=(sf.e_ff/sf.t_ff(z_val))*m_g
    return f_m_star

def diff_eqn_zgas_2(t,y,m_g,m_d_cg,m_halo,m_d_s_d,z_star):      #diff_eqns_2
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_d_cg)-(y*(sf.e_ff/sf.t_ff(z_val)))+(y_z*m_d_s_d)-(snw.eta(z_val,m_halo,z_star)*(y/m_g)*m_d_s_d)
    return f_m_z_gas

#INSTANTANEOUS FEEDBACK

def diff_eqn_eq_gas_1(t,y,m_d_cg,m_halo,z_star):        #diff_eqns_eq_1
    z_val=z(t)
    
    f_m_gas=m_d_cg-(sf.e_ff/sf.t_ff(z_val))*y-snw.eta(z_val,m_halo,z_star)*(sf.e_ff/sf.t_ff(z_val))*y
    return f_m_gas

def diff_eqn_eq_star_1(t,y,m_g):       #diff_eqns_eq_1
    z_val=z(t)
    
    f_m_star=(sf.e_ff/sf.t_ff(z_val))*m_g
    return f_m_star

def diff_eqn_eq_zgas_1(t,y,m_g,m_d_cg,m_halo,z_star):        #diff_eqns_eq_1
    z_val=z(t)
    f_m_star=(sf.e_ff/sf.t_ff(z_val))*m_g
    
    f_m_z_gas=(z_igm*m_d_cg)-(snw.eta(z_val,m_halo,z_star)*y*sf.e_ff/sf.t_ff(z_val))+(y_z*f_m_star)-(y*sf.e_ff/sf.t_ff(z_val))
    return f_m_z_gas



# STELLAR METALLICITY EQUATIONS- Remove (y_z*e_ff/t_ff(z_val)*m_g) if not needed

def diff_eqn_zstar_1(t,y,m_g):
    z_val=z(t)
    f_m_z_star=0.0
    return f_m_z_star

def diff_eqn_zstar_2(t,y,m_g,m_z_g):
    z_val=z(t)
    f_m_z_star=(m_z_g)*(sf.e_ff/sf.t_ff(z_val))
    return f_m_z_star


start=30
stop=50
check=86

no=10        #HALO MASS POWER VALUE

uv_choice=input("Do you want to include background UV suppression or not?")



#DELAYED FEEDBACK

for i in range(start,stop,1):
    redshift=np.array([])
    halo_mass=np.array([])
    halo_mass_rate=np.array([])
    print(i)

    redshift=np.loadtxt(f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Data Sets/Sorted Data/mh{no}_data/Redshifts/redshift_{i}.txt",delimiter=' ')
    halo_mass=np.loadtxt(f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Data Sets/Sorted Data/mh{no}_data/Halo Mass/halo_mass_{i}.txt",delimiter=' ')
    halo_mass_rate=np.loadtxt(f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Data Sets/Sorted Data/mh{no}_data/Halo Mass Rate/halo_mass_rate_{i}.txt",delimiter=' ')
    
    cosmic_time=t(redshift)
    h=(cosmic_time[len(cosmic_time)-1]-cosmic_time[0])/len(cosmic_time)
    
    print(len(cosmic_time))
    
    
    
    tsn=cosmic_time[0]+t_d        #Also a varying parameter

    
    if (uv_choice == 'Yes' or uv_choice == 'yes'):
        m_dot_cg_val=m_dot_cg_with_UV(redshift,halo_mass,halo_mass_rate)
 
    elif (uv_choice == 'No' or uv_choice == 'no'):
        m_dot_cg_val=m_dot_cg_no_UV(redshift,halo_mass,halo_mass_rate)

    
    ini_m_gas=[0.0]
    ini_m_star=[0.0]
    
    ini_m_z_gas=[0.0]
    ini_m_z_star=[0.0]
    
    ini_m_dust=[0.0]
    
    m_g_val_1=[]
    m_star_val_1=[]
    m_z_g_val_1=[]
    m_z_star_val_1=[]
    m_dust_val_1=[]
    
    m_dot_star_vals=[]
    
    z_star_val=0.0
    
    
    f_vals=[]
    
    k=0

    for j in range(0,len(cosmic_time)):
        
        if (j == 0):
            t_span=[cosmic_time[j],cosmic_time[j]]
        else:
            t_span=[cosmic_time[j-1],cosmic_time[j]]
        
        if (cosmic_time[j] <= tsn):
            
            solution=solve_ivp(diff_eqn_gas_1,t_span,ini_m_gas,args=[m_dot_cg_val[j]],max_step=h)
            m_g = solution.y[0][len(solution.y[0])-1]
            
            solution=solve_ivp(diff_eqn_star_1,t_span,ini_m_star,args=[ini_m_gas[0]],max_step=h)
            m_s = solution.y[0][len(solution.y[0])-1]
            
            solution=solve_ivp(diff_eqn_zgas_1,t_span,ini_m_z_gas,args=[ini_m_gas[0],m_dot_cg_val[j]],max_step=h)
            m_z_g = solution.y[0][len(solution.y[0])-1]
                
            solution=solve_ivp(diff_eqn_zstar_2,t_span,ini_m_z_star,args=[ini_m_gas[0],ini_m_z_gas[0]],max_step=h)
            m_z_s = solution.y[0][len(solution.y[0])-1]
            
            
        elif (cosmic_time[j] > tsn):
            
            solution=solve_ivp(diff_eqn_gas_2,t_span,ini_m_gas,args=[m_dot_cg_val[j],halo_mass[j],m_dot_star_vals[k],z_star_val],max_step=h)
            m_g = solution.y[0][len(solution.y[0])-1]
            
            solution=solve_ivp(diff_eqn_star_2,t_span,ini_m_star,args=[ini_m_gas[0]],max_step=h)
            m_s = solution.y[0][len(solution.y[0])-1]
            
            solution=solve_ivp(diff_eqn_zgas_2,t_span,ini_m_z_gas,args=[ini_m_gas[0],m_dot_cg_val[j],halo_mass[j],m_dot_star_vals[k],z_star_val],max_step=h)
            m_z_g = solution.y[0][len(solution.y[0])-1]
                
            solution=solve_ivp(diff_eqn_zstar_2,t_span,ini_m_z_star,args=[ini_m_gas[0],ini_m_z_gas[0]],max_step=h)
            m_z_s = solution.y[0][len(solution.y[0])-1]
            
            
            k=k+1
        
        ini_m_gas=[m_g]
        ini_m_star=[m_s]
        ini_m_z_gas=[m_z_g]
        ini_m_z_star=[m_z_s]
        ini_m_dust=[m_d]
        
        if (ini_m_gas[0] < 0.0):
            ini_m_gas[0]=0.0
            ini_m_z_gas[0]=0.0
        
        if (ini_m_star[0] < 0.0):
            ini_m_star[0]=0.0
            ini_m_z_star[0]=0.0
            
        if (ini_m_z_gas[0] < 0.0):
            ini_m_z_gas[0]=0.0
            
        if (ini_m_z_star[0] < 0.0):
            ini_m_z_star[0]=0.0   

        if (ini_m_dust[0] < 0.0):
            ini_m_dust[0]=0.0

        if (ini_m_star[0] == 0.0):
            z_star_val=0.0
            
        elif (ini_m_star[0] > 0.0):
            z_star_val=ini_m_z_star[0]/ini_m_star[0]
        
            
        m_dot_star_val=(e_ff/t_ff(redshift[j]))*ini_m_gas[0]
        m_dot_star_vals=np.append(m_dot_star_vals,[m_dot_star_val])

        f_val=f_sigmoid(z_star_val)                 #TEMP
        f_vals=np.append(f_vals,f_val)              #TEMP

        m_g_val_1=np.append(m_g_val_1,ini_m_gas)
        m_star_val_1=np.append(m_star_val_1,ini_m_star)
        m_z_g_val_1=np.append(m_z_g_val_1,ini_m_z_gas)
        m_z_star_val_1=np.append(m_z_star_val_1,ini_m_z_star)
        m_dust_val_1=np.append(m_dust_val_1,ini_m_dust)
    
    print(len(m_g_val_1))
    print(m_z_star_val_1)
    
    '''
    np.savetxt(f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/gas/tree_{i}.txt",m_g_val_1,delimiter=' ')
    np.savetxt(f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/star/tree_{i}.txt",m_star_val_1,delimiter=' ')
    np.savetxt(f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/z_gas/tree_{i}.txt",m_z_g_val_1,delimiter=' ')
    np.savetxt(f"C:/Users/Anand Menon/Documents/ICRAR Stuff/Results/No UV Delay/mh{no}_data/z_star/tree_{i}.txt",m_z_star_val_1,delimiter=' ')
    
    
    plt.semilogy(cosmic_time,halo_mass,label='Halo mass')
    plt.semilogy(cosmic_time,halo_mass_rate,label='Halo mass rate')
        
    plt.semilogy(cosmic_time,m_g_val_1,label='Gas mass values')
    plt.semilogy(cosmic_time,m_star_val_1,label='Stellar mass values')
    
    plt.semilogy(cosmic_time,m_z_g_val_1,label='Gas metallicity values')
    plt.semilogy(cosmic_time,m_z_star_val_1,label='Stellar metallicity')
    
    plt.semilogy(cosmic_time,m_dust_val_1,label='Dust mass')
    
    plt.ylim(10**-3,10**10)
    '''
    plt.plot(cosmic_time,f_vals,label='Stellar metallicity fraction')
    
#plt.legend() 
plt.ylim(0.2,0.8)
plt.xlim(0.18,1.2)   

plt.xlabel('Cosmic Time')
plt.ylabel('f($z_\star$)')


plt.show()



#DELAYED FEEDBACK MODELS


def delayed_feedback_uv(m_h0_val,e_ff,epsilon_p,gamma_ff):    #DELAYED FEEDBACK + UV
    
    m_g_val_1=[]
    m_star_val_1=[]
    m_z_g_val_1=[]
    m_z_star_val_1=[]

    m_dot_star_vals=[]

    i=0
    j=0

    tsn=0.015+t0

    values=np.array([0.0,0.0])
    m_z_gas=[0.0]
    m_z_star=[0.0]
    
    for t in cosmic_time:
        redshift=z(t)
        
        if (i == 0):
            t_span=[t,t]
        else:
            t_span=[cosmic_time[i-1],t]
        
        if t <= tsn:
            
            k1=h*diff_eqns_1(t,values,m_h0_val,e_ff,gamma_ff)
            k2=h*diff_eqns_1(t+h/2,values+k1/2,m_h0_val,e_ff,gamma_ff)
            k3=h*diff_eqns_1(t+h/2,values+k2/2,m_h0_val,e_ff,gamma_ff)
            k4=h*diff_eqns_1(t+h,values+k3,m_h0_val,e_ff,gamma_ff)
            
            if (m_z_gas[0] == 0.0):
                solution=solve_ivp(diff_eqn_zgas_1,t_span,m_z_gas,args=[m_h0_val],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]
                
                solution=solve_ivp(diff_eqn_zstar_1,t_span,m_z_star,max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]

            else:
                solution=solve_ivp(diff_eqn_zgas_2,t_span,m_z_gas,args=[values[0],m_h0_val,e_ff,gamma_ff],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]
                
                
                solution=solve_ivp(diff_eqn_zstar_2,t_span,m_z_star,args=[m_z_gas[0],m_h0_val,e_ff,gamma_ff],max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]
            
        elif t > tsn:
            k1=h*diff_eqns_2(t,values,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            k2=h*diff_eqns_2(t+h/2,values+k1/2,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            k3=h*diff_eqns_2(t+h/2,values+k2/2,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            k4=h*diff_eqns_2(t+h,values+k3,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            
            if (m_z_gas[0] == 0.0):
                solution=solve_ivp(diff_eqn_zgas_3,t_span,m_z_gas,args=[m_h0_val,m_dot_star_vals[j]],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]
                
                solution=solve_ivp(diff_eqn_zstar_1,t_span,m_z_star,max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]

            else:
                solution=solve_ivp(diff_eqn_zgas_4,t_span,m_z_gas,args=[values[0],m_h0_val,m_dot_star_vals[j],e_ff,gamma_ff,epsilon_p],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]
                
                solution=solve_ivp(diff_eqn_zstar_2,t_span,m_z_star,args=[m_z_gas[0],m_h0_val,e_ff,gamma_ff],max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]
            
            j=j+1
            
        values=values+(k1+2*k2+2*k3+k4)/6
        m_z_gas=[m_z_g]
        m_z_star=[m_z_s]
        
        i=i+1
        
        if (values[0] < 0.0):
            values[0]=0.0
            m_z_gas[0]=0.0    
        
        if (values[1] < 0.0):
            values[1]=0.0
            m_z_star[0]=0.0
            
        if (m_z_gas[0] < 0.0):
            m_z_gas[0]=0.0
         
        if (m_z_star[0] < 0.0):
            m_z_star[0]=0.0
          
        m_dot_star_val=(e_ff/sf.t_ff(redshift))*values[0]
        m_dot_star_vals=np.append(m_dot_star_vals,[m_dot_star_val])
        
        m_g_val_1=np.append(m_g_val_1,[values[0]])
        m_star_val_1=np.append(m_star_val_1,[values[1]])
        m_z_g_val_1=np.append(m_z_g_val_1,m_z_gas)
        m_z_star_val_1=np.append(m_z_star_val_1,m_z_star)
    
    return m_g_val_1,m_star_val_1,m_z_g_val_1,m_z_star_val_1    
    


def delayed_feedback(m_h0_val,e_ff,epsilon_p,gamma_ff):    #DELAYED FEEDBACK + NO UV
    
    m_g_val_1=[]
    m_star_val_1=[]
    m_z_g_val_1=[]
    m_z_star_val_1=[]

    m_dot_star_vals=[]

    i=0
    j=0

    tsn=0.015+t0

    values=np.array([0.0,0.0])
    m_z_gas=[0.0]
    m_z_star=[0.0]
     
    for t in cosmic_time:
        redshift=z(t)
        
        if (i == 0):
            t_span=[t,t]
        else:
            t_span=[cosmic_time[i-1],t]
        
        if t <= tsn:
            
            k1=h*diff_eqns_3(t,values,m_h0_val,e_ff,gamma_ff)
            k2=h*diff_eqns_3(t+h/2,values+k1/2,m_h0_val,e_ff,gamma_ff)
            k3=h*diff_eqns_3(t+h/2,values+k2/2,m_h0_val,e_ff,gamma_ff)
            k4=h*diff_eqns_3(t+h,values+k3,m_h0_val,e_ff,gamma_ff)
               
            if (m_z_gas[0] == 0.0):
                solution=solve_ivp(diff_eqn_zgas_5,t_span,m_z_gas,args=[m_h0_val],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]

                solution=solve_ivp(diff_eqn_zstar_1,t_span,m_z_star,max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]

            else:
                solution=solve_ivp(diff_eqn_zgas_6,t_span,m_z_gas,args=[values[0],m_h0_val,e_ff,gamma_ff],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]
                
                solution=solve_ivp(diff_eqn_zstar_2,t_span,m_z_star,args=[m_z_gas[0],m_h0_val,e_ff,gamma_ff],max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]
            
        elif t > tsn:
            k1=h*diff_eqns_4(t,values,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            k2=h*diff_eqns_4(t+h/2,values+k1/2,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            k3=h*diff_eqns_4(t+h/2,values+k2/2,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            k4=h*diff_eqns_4(t+h,values+k3,m_dot_star_vals[j],m_h0_val,e_ff,epsilon_p,gamma_ff)
            
            if (m_z_gas[0] == 0.0):
                solution=solve_ivp(diff_eqn_zgas_7,t_span,m_z_gas,args=[m_h0_val,m_dot_star_vals[j]],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]
                
                solution=solve_ivp(diff_eqn_zstar_1,t_span,m_z_star,max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]
            else:
                solution=solve_ivp(diff_eqn_zgas_8,t_span,m_z_gas,args=[values[0],m_h0_val,m_dot_star_vals[j],e_ff,gamma_ff,epsilon_p],max_step=h)
                m_z_g = solution.y[0][len(solution.y[0])-1]

                solution=solve_ivp(diff_eqn_zstar_2,t_span,m_z_star,args=[m_z_gas[0],m_h0_val,e_ff,gamma_ff],max_step=h)
                m_z_s = solution.y[0][len(solution.y[0])-1]
            
            j=j+1
            
        values=values+(k1+2*k2+2*k3+k4)/6
        m_z_gas=[m_z_g]
        m_z_star=[m_z_s]
        
        i=i+1
        
        if (values[0] < 0.0):
            values[0]=0.0
            m_z_gas[0]=0.0    
        
        if (values[1] < 0.0):
            values[1]=0.0
            m_z_star[0]=0.0
            
        if (m_z_gas[0] < 0.0):
            m_z_gas[0]=0.0
         
        if (m_z_star[0] < 0.0):
            m_z_star[0]=0.0
          
        m_dot_star_val=(e_ff/sf.t_ff(redshift))*values[0]
        m_dot_star_vals=np.append(m_dot_star_vals,[m_dot_star_val])
        
        m_g_val_1=np.append(m_g_val_1,[values[0]])
        m_star_val_1=np.append(m_star_val_1,[values[1]])
        m_z_g_val_1=np.append(m_z_g_val_1,m_z_gas)
        m_z_star_val_1=np.append(m_z_star_val_1,m_z_star)
    
    return m_g_val_1,m_star_val_1,m_z_g_val_1,m_z_star_val_1

#INSTANTANEOUS FEEDBACK MODEL

def instantaneous_feedback_uv(m_h0_val,e_ff,epsilon_p,gamma_ff):    #INSTANTANEOUS FEEDBACK + UV
    m_g_eq_val_1=[]
    m_star_eq_val_1=[]
    m_z_g_eq_val_1=[]
    m_z_star_eq_val_1=[]

    i=0

    values=np.array([0.0,0.0])
    m_z_gas=[0.0]
    m_z_star=[0.0]
    
    for t in cosmic_time:
        redshift=z(t)
        
        if (i == 0):
            t_span=[t,t]
        else:
            t_span=[cosmic_time[i-1],t]
        
        k1=h*diff_eqns_eq_1(t,values,m_h0_val,e_ff,epsilon_p,gamma_ff)
        k2=h*diff_eqns_eq_1(t+h/2,values+k1/2,m_h0_val,e_ff,epsilon_p,gamma_ff)
        k3=h*diff_eqns_eq_1(t+h/2,values+k2/2,m_h0_val,e_ff,epsilon_p,gamma_ff)
        k4=h*diff_eqns_eq_1(t+h,values+k3,m_h0_val,e_ff,epsilon_p,gamma_ff)
        
        if (m_z_gas[0] == 0.0):
            solution=solve_ivp(diff_eqn_eq_zgas_1,t_span,m_z_gas,args=[values[0],m_h0_val,e_ff,gamma_ff],max_step=h)
            m_z_g=solution.y[0][len(solution.y[0])-1]

            solution=solve_ivp(diff_eqn_zstar_1,t_span,m_z_star,max_step=h)
            m_z_s=solution.y[0][len(solution.y[0])-1]

        else:
            solution=solve_ivp(diff_eqn_eq_zgas_2,t_span,m_z_gas,args=[values[0],m_h0_val,e_ff,gamma_ff,epsilon_p],max_step=h)
            m_z_g=solution.y[0][len(solution.y[0])-1]
            
            solution=solve_ivp(diff_eqn_zstar_2,t_span,m_z_star,args=[m_z_gas[0],m_h0_val,e_ff,gamma_ff],max_step=h)
            m_z_s=solution.y[0][len(solution.y[0])-1]
        
        values=values+(k1+2*k2+2*k3+k4)/6
        m_z_gas=[m_z_g]
        m_z_star=[m_z_s]
        
        i=i+1
        
        if (values[0] < 0.0):
            values[0]=0.0
            m_z_gas[0]=0.0    
        
        if (values[1] < 0.0):
            values[1]=0.0
            m_z_star[0]=0.0
            
        if (m_z_gas[0] < 0.0):
            m_z_gas[0]=0.0
            
        if (m_z_star[0] < 0.0):
            m_z_star[0]=0.0
            
         
        m_g_eq_val_1=np.append(m_g_eq_val_1,[values[0]])
        m_star_eq_val_1=np.append(m_star_eq_val_1,[values[1]])
        m_z_g_eq_val_1=np.append(m_z_g_eq_val_1,m_z_gas)
        m_z_star_eq_val_1=np.append(m_z_star_eq_val_1,m_z_star)
        
    return m_g_eq_val_1,m_star_eq_val_1,m_z_g_eq_val_1,m_z_star_eq_val_1
 
def instantaneous_feedback(m_h0_val,e_ff,epsilon_p,gamma_ff):    #INSTANTANEOUS FEEDBACK + NO UV
    m_g_eq_val_1=[]
    m_star_eq_val_1=[]
    m_z_g_eq_val_1=[]
    m_z_star_eq_val_1=[]

    i=0

    values=np.array([0.0,0.0])
    m_z_gas=[0.0]
    m_z_star=[0.0]
    
    for t in cosmic_time:
        redshift=z(t)
        
        if (i == 0):
            t_span=[t,t]
        else:
            t_span=[cosmic_time[i-1],t]
        
        k1=h*diff_eqns_eq_2(t,values,m_h0_val,e_ff,epsilon_p,gamma_ff)
        k2=h*diff_eqns_eq_2(t+h/2,values+k1/2,m_h0_val,e_ff,epsilon_p,gamma_ff)
        k3=h*diff_eqns_eq_2(t+h/2,values+k2/2,m_h0_val,e_ff,epsilon_p,gamma_ff)
        k4=h*diff_eqns_eq_2(t+h,values+k3,m_h0_val,e_ff,epsilon_p,gamma_ff)
        
        if (m_z_gas[0] == 0.0):
            solution=solve_ivp(diff_eqn_eq_zgas_3,t_span,m_z_gas,args=[values[0],m_h0_val,e_ff,gamma_ff],max_step=h)
            m_z_g=solution.y[0][len(solution.y[0])-1]
            
            solution=solve_ivp(diff_eqn_zstar_1,t_span,m_z_star,max_step=h)
            m_z_s=solution.y[0][len(solution.y[0])-1]

        else:
            solution=solve_ivp(diff_eqn_eq_zgas_4,t_span,m_z_gas,args=[values[0],m_h0_val,e_ff,gamma_ff,epsilon_p],max_step=h)
            m_z_g=solution.y[0][len(solution.y[0])-1]
            
            solution=solve_ivp(diff_eqn_zstar_2,t_span,m_z_star,args=[m_z_gas[0],m_h0_val,e_ff,gamma_ff],max_step=h)
            m_z_s=solution.y[0][len(solution.y[0])-1]
        
        values=values+(k1+2*k2+2*k3+k4)/6
        m_z_gas=[m_z_g]
        m_z_star=[m_z_s]
        
        i=i+1
        
        if (values[0] < 0.0):
            values[0]=0.0
            m_z_gas[0]=0.0    
        
        if (values[1] < 0.0):
            values[1]=0.0
            m_z_star[0]=0.0
            
        if (m_z_gas[0] < 0.0):
            m_z_gas[0]=0.0
            
        if (m_z_star[0] < 0.0):
            m_z_star[0]=0.0
            
            
        m_g_eq_val_1=np.append(m_g_eq_val_1,[values[0]])
        m_star_eq_val_1=np.append(m_star_eq_val_1,[values[1]])
        m_z_g_eq_val_1=np.append(m_z_g_eq_val_1,m_z_gas)
        m_z_star_eq_val_1=np.append(m_z_star_eq_val_1,m_z_star)
        
    return m_g_eq_val_1,m_star_eq_val_1,m_z_g_eq_val_1,m_z_star_eq_val_1 



#PARAMETERS

#mh_vals=np.array([10**6,10**7,10**8,10**9])
mh_vals=10**7

e_ff_vals=np.array([0.1,0.015,0.0015])
#e_ff_vals=0.015

#epsilon_p_vals=np.array([2,5,7])
epsilon_p_vals=5

#gamma_ff_vals=np.array([0.111,0.141])
gamma_ff_vals=0.141

#PLOTTING FOR EVOLUTION OF MASSES WITH COSMIC TIME

fig, axs=plt.subplots(4,4,figsize=(30,20))
fig.tight_layout()
alpha=0.55
lw=1.75

colors=['navy','green','maroon','darkmagenta']

for i in range(0,3):
    print(i)
    m_g_val_1,m_star_val_1,m_z_g_val_1,m_z_star_val_1=instantaneous_feedback(mh_vals,e_ff_vals[i],epsilon_p_vals,gamma_ff_vals)
    m_g_val_2,m_star_val_2,m_z_g_val_2,m_z_star_val_2=instantaneous_feedback_uv(mh_vals,e_ff_vals[i],epsilon_p_vals,gamma_ff_vals)
    m_g_val_3,m_star_val_3,m_z_g_val_3,m_z_star_val_3=delayed_feedback(mh_vals,e_ff_vals[i],epsilon_p_vals,gamma_ff_vals)
    m_g_val_4,m_star_val_4,m_z_g_val_4,m_z_star_val_4=delayed_feedback_uv(mh_vals,e_ff_vals[i],epsilon_p_vals,gamma_ff_vals)
    
    if (i == 0):
        axs[0,0].semilogy(cosmic_time,m_g_val_1,color=f'{colors[i]}',linewidth=lw,label='$e_{ff}=0.1$',alpha=alpha)
        axs[0,1].semilogy(cosmic_time,m_g_val_2,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,2].semilogy(cosmic_time,m_g_val_3,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,3].semilogy(cosmic_time,m_g_val_4,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    elif (i == 1):
        axs[0,0].semilogy(cosmic_time,m_g_val_1,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,1].semilogy(cosmic_time,m_g_val_2,color=f'{colors[i]}',linewidth=lw,label='$e_{ff}=0.015$',alpha=alpha)
        axs[0,2].semilogy(cosmic_time,m_g_val_3,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,3].semilogy(cosmic_time,m_g_val_4,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    elif (i == 2):
        axs[0,0].semilogy(cosmic_time,m_g_val_1,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,1].semilogy(cosmic_time,m_g_val_2,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,2].semilogy(cosmic_time,m_g_val_3,color=f'{colors[i]}',linewidth=lw,label='$e_{ff}=0.0015$',alpha=alpha)
        axs[0,3].semilogy(cosmic_time,m_g_val_4,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    """
    elif (i == 3):
        axs[0,0].semilogy(cosmic_time,m_g_val_1,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,1].semilogy(cosmic_time,m_g_val_2,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,2].semilogy(cosmic_time,m_g_val_3,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
        axs[0,3].semilogy(cosmic_time,m_g_val_4,color=f'{colors[i]}',linewidth=lw,label='$m_{h}=10^9M_{\odot}$',alpha=alpha)
    """
    
    axs[1,0].semilogy(cosmic_time,m_star_val_1,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[2,0].semilogy(cosmic_time,m_z_g_val_1,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[3,0].semilogy(cosmic_time,m_z_star_val_1,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    
    axs[1,1].semilogy(cosmic_time,m_star_val_2,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[2,1].semilogy(cosmic_time,m_z_g_val_2,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[3,1].semilogy(cosmic_time,m_z_star_val_2,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    
    axs[1,2].semilogy(cosmic_time,m_star_val_3,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[2,2].semilogy(cosmic_time,m_z_g_val_3,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[3,2].semilogy(cosmic_time,m_z_star_val_3,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    
    axs[1,3].semilogy(cosmic_time,m_star_val_4,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[2,3].semilogy(cosmic_time,m_z_g_val_4,color=f'{colors[i]}',linewidth=lw,alpha=alpha)
    axs[3,3].semilogy(cosmic_time,m_z_star_val_4,color=f'{colors[i]}',linewidth=lw,alpha=alpha)

    
axs[0,0].set_title('Instantaneous feedback only (no UV)',fontsize=20)
axs[0,0].set_ylabel('$m_{g}$ $(M_{\odot})$',fontsize=20)
axs[0,0].legend(prop={"size":20})

axs[1,0].set_ylabel('$m_{\star}$ $(M_{\odot})$',fontsize=20)

axs[2,0].set_ylabel('$m_{Z,g}$ $(M_{\odot})$',fontsize=20)


axs[3,0].set_ylabel('$m_{Z,\star}$ $(M_{\odot})$',fontsize=20)
axs[1,0].set_xlabel('Cosmic Time (in Gyr)',fontsize=20)

axs[0,1].set_title('Instantaneous feedback + UV',fontsize=20)
axs[0,1].legend(prop={"size":20})


axs[1,1].set_xlabel('Cosmic Time (in Gyr)',fontsize=20)

axs[0,2].set_title('Delayed feedback only (no UV)',fontsize=20)
axs[0,2].legend(prop={"size":20})

axs[1,2].set_xlabel('Cosmic Time (in Gyr)',fontsize=20)

axs[0,3].set_title('Delayed feedback + UV',fontsize=20)
axs[0,3].legend(prop={"size":20})

axs[1,3].set_xlabel('Cosmic Time (in Gyr)',fontsize=20)


for i in range(0,4):
    for j in range (0,4):
        axs[i,j].tick_params(width=2)
        axs[i,j].grid(ls='--')
        axs[i,j].set_xlim([0,2])
        
        if (i == 0 or i == 1):
            axs[i,j].set_ylim([10**(1),10**9])
        
        elif (i == 2 or i ==3):
            axs[i,j].set_ylim([10**(-2),10**6])
        
        for axis in ['top','bottom','left','right']:
            axs[i,j].spines[axis].set_linewidth(2)

for ax in axs.flat:
    ax.label_outer()
    ax.tick_params(axis='both',direction='in',labelsize=15,which='both')

plt.show()


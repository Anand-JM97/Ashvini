# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:47:28 2024

@author: Anand Menon
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import units as cu
from astropy.cosmology import z_at_value
import astropy.units as u

H_0=cosmo.H0 #in km / (Mpc s)
H_0=H_0.to(u.Gyr**(-1)) #in 1/Gyr

A=0.030

omega_m=cosmo.Om0
omega_b=cosmo.Ob0
omega_L=cosmo.Ode0



alfa=0.79
pi=np.pi

plt.rcParams['figure.dpi'] = 300

#t=cosmo.age(z)

#z=z_at_value(cosmo.age,t)

def t(z):
    t_val=cosmo.age(z)
    return t_val

def z(t):
    z_val=z_at_value(cosmo.age,t)
    return z_val

def H(z):
    H=cosmo.H(z)
    H=H.to(u.Gyr**(-1)) #in 1/Gyr
    return H

def evolve_galaxies():
    """This will be the main function. We should have this call star formation, supernovae feedback, and such defined
    in the other files. Ideally this should be the only the function that is defined in this file.
    """

    return 1


#Initialization for the halo mass evolution
m_h0=np.logspace(7,10,num=20)

#Quantities for suppression due to UV heating 

z_rei=7
gamma=15
omega=2
c_omega=2**(omega/3)-1
beta=z_rei*((np.log(1.82*(10**3)*np.exp(-0.63*z_rei)-1))**(-1/gamma))
t0=0.129
tf=5
n=25000
h=(tf-t0)/n
cosmic_time=np.linspace(t0,tf,n)

#Star formation parameters

R=0

#Metallicity parameters

z_igm=10**(-3)
y_z=0.06
zeta_w=1

def z(t):
    """
    Function to convert cosmic time to redshift.
    Args:
        t (float): Parameter representing cosmic time.

    Returns:
        Float: The redshift value
    """
    k_val=2/(3*H_0*np.sqrt(omega_m))
    z_0=25
    t_0=0.129
    term_1=1/((1+z_0)**(3/2))
    z_val=((term_1+(t-t_0)/k_val)**(-2/3))-1
    return z_val

def H(z):
    """
    Hubble function.
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The value of the Hubble constant for the redshift value entered as the argument.
    """
    H=H_0*np.sqrt(omega_m*(1+z)**3+omega_L)
    return H

def M_c(z):
    """
    Characteristic mass scale for reionization (Okamoto et al. 2008).
    Args:
        z (float): Parameter for redshift.

    Returns:
        Float: The characteristic mass scale at which the baryon fraction is suppressed by a factor of two, compared to the universal value, because of background UV.
    """
    M_val=1.69*(10**10)*(np.exp(-0.63*z)/(1+np.exp((z/beta)**gamma)))
    return M_val

def s(x,y):
    """
    A step function used in the expression for accretion suppression due to UV background.
    Args:
        x, y (float): Two parameters representing any physical quantity.

    Returns:
        The value of the function s(x,y) based on the argument of the parameters.
    """
    s_val=(1+(2**(y/3)-1)*(x**(-y)))**(-3/y)
    return s_val

def delta_c(z):
    """
    
    """
    d=((omega_m*((1+z)**3))/((omega_m*((1+z)**3))+omega_L))-1
    delta_c_val=18*(pi**2)+(82*d)-(39*(d**2))
    return delta_c_val

def mdot_h(z,m_h0_value):
    """
    Function that tracks the Halo growth rate.
    Args:
        z (float): Redshift parameter.
        m_h0_value (float): 
    """
    z_0=5
    halo_rate=A*m_h0_value*np.exp(-alfa*(z-z_0))*(1+z)**(5/2)
    return halo_rate

def m_h(z,m_h0_value):
    z_0=5
    m_h_val=m_h0_value*np.exp(-alfa*(z-z_0))
    return m_h_val

def mu_c(z,m_halo):
    mu=m_h(z,m_halo)/M_c(z)
    return mu

def X(z,m_halo):
    M_omega=(M_c(z)/m_h(z,m_halo))**(omega)
    X_val=(3*c_omega*M_omega)/(1+c_omega*M_omega)
    return X_val

def epsilon(z):
    part2=(gamma*(z**(gamma-1))/(beta**gamma))*(np.exp((z/beta)**gamma))/((1+np.exp((z/beta)**gamma))**2)
    epsilon=(0.63)/(1+np.exp((z/beta)**gamma))+part2
    return epsilon

def epsilon2(z):
    part2=(gamma*(z**(gamma-1))/beta**gamma)*(np.exp((z/beta)**gamma))/(1+np.exp((z/beta)**gamma))
    epsilon2_val=(0.63)+part2
    return epsilon2_val

def g(z,m_halo):
    m_vir=m_h(z,m_halo)
    c_vir=15*((m_vir/(10**12))**(-0.2))/(1+z)
    A=np.log(1+c_vir)-(c_vir)/(1+c_vir)
    g_val=(A/(delta_c(z)*c_vir))**(0.5)
    return g_val

def t_ff(z,m_halo,gamma_ff):
    t_ff_val=gamma_ff/(H_0*np.sqrt((omega_m*((1+z)**3))+omega_L))
    return t_ff_val
    
def eta(z,m_halo,epsilon_p):
    pi_fid=1
    #epsilon_p=5
    
    eta_p=epsilon_p*pi_fid*(((10**11.5)/m_h(z,m_halo))**(1/3))*((9/(1+z))**(1/2))

    return eta_p

def m_dot_star(m_gas,z,m_halo,e_ff,gamma_ff):
    m_dot_star_val=(e_ff*m_gas/t_ff(z,m_halo,gamma_ff))
    return m_dot_star_val

def m_dot_wind(m_gas,z,m_halo,epsilon_p):
    m_dot_wind_val=eta(z,m_halo,epsilon_p)*m_dot_star(m_gas,z,m_halo)
    return m_dot_wind_val

def epsilon_uv_1(z_val,m_h0_val):
    if (z_val > 10):
        value=1
    else:
        value=s(mu_c(z_val,m_h0_val),omega)*((1+X(z_val,m_h0_val))-2*epsilon(z_val)*m_h(z_val,m_h0_val)*X(z_val,m_h0_val)*(1+z_val)*H(z_val)/mdot_h(z_val,m_h0_val))
    if (value < 0):
        value=0
    return value

def m_dot_cg(z,m_h0_val):
    m_dot_cg_val=(omega_b/omega_m)*mdot_h(z,m_h0_val)*epsilon_uv_1(z,m_h0_val)
    return m_dot_cg_val

def m_dot_cg_2(z,m_h0_val):
    m_dot_cg_val=(omega_b/omega_m)*mdot_h(z,m_h0_val)
    return m_dot_cg_val

def diff_eqns_1(t,r,m_h0_val,e_ff,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg(z_val,m_h0_val)-(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    
        
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_2(t,r,m_d_s_d,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg(z_val,m_h0_val)-(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g-eta(z_val,m_h0_val,epsilon_p)*m_d_s_d
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    
   
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_3(t,r,m_h0_val,e_ff,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg_2(z_val,m_h0_val)-(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
        
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_4(t,r,m_d_s_d,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)
    
    f_m_g=m_dot_cg_2(z_val,m_h0_val)-(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g-eta(z_val,m_h0_val,epsilon_p)*m_d_s_d
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
        
    return(np.array([f_m_g,f_m_star]))

def diff_eqns_eq_1(t,r,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)

    f_m_g=m_dot_cg(z_val,m_h0_val)-(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g-eta(z_val,m_h0_val,epsilon_p)*(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
            
    return(np.array([f_m_g,f_m_star]))    

def diff_eqns_eq_2(t,r,m_h0_val,e_ff,epsilon_p,gamma_ff):
    m_g=r[0]
    m_star=r[1]
    
    z_val=z(t)

    f_m_g=m_dot_cg_2(z_val,m_h0_val)-(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g-eta(z_val,m_h0_val,epsilon_p)*(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    
    return(np.array([f_m_g,f_m_star]))    

def diff_eqn_zgas_1(t,y,m_h0_val):  #diff_eqns_1
    z_val=z(t)
    
    f_m_z_gas=z_igm*m_dot_cg(z_val,m_h0_val)
    return f_m_z_gas

def diff_eqn_zgas_2(t,y,m_g,m_h0_val,e_ff,gamma_ff):    #diff_eqns_1
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))-((y/m_g)*(1-R)*(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g)
    return f_m_z_gas

def diff_eqn_zgas_3(t,y,m_h0_val,m_d_s_d):      #diff_eqns_2
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))+(y_z*m_d_s_d)
    return f_m_z_gas

def diff_eqn_zgas_4(t,y,m_g,m_h0_val,m_d_s_d,e_ff,gamma_ff,epsilon_p):      #diff_eqns_2
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))-(y*(1-R)*(e_ff/t_ff(z_val,m_h0_val,gamma_ff)))+(y_z*m_d_s_d)-(eta(z_val,m_h0_val,epsilon_p)*(y/m_g)*m_d_s_d)
    return f_m_z_gas

def diff_eqn_zgas_5(t,y,m_h0_val):       #diff_eqns_3
    z_val=z(t)

    f_m_z_gas=z_igm*m_dot_cg_2(z_val,m_h0_val)
    return f_m_z_gas

def diff_eqn_zgas_6(t,y,m_g,m_h0_val,e_ff,gamma_ff):      #diff_eqns_3
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))-((y/m_g)*(1-R)*(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g)
    return f_m_z_gas

def diff_eqn_zgas_7(t,y,m_h0_val,m_d_s_d):       #diff_eqns_4
    z_val=z(t)

    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))+(y_z*m_d_s_d)
    return f_m_z_gas

def diff_eqn_zgas_8(t,y,m_g,m_h0_val,m_d_s_d,e_ff,gamma_ff,epsilon_p):       #diff_eqns_4
    z_val=z(t)
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))-(y*(1-R)*(e_ff/t_ff(z_val,m_h0_val,gamma_ff)))+(y_z*m_d_s_d)-(eta(z_val,m_h0_val,epsilon_p)*(y/m_g)*m_d_s_d)
    return f_m_z_gas

def diff_eqn_eq_zgas_1(t,y,m_g,m_h0_val,e_ff,gamma_ff):       #diff_eqns_eq_1
    z_val=z(t)
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))+(y_z*f_m_star)
    return f_m_z_gas
    
def diff_eqn_eq_zgas_2(t,y,m_g,m_h0_val,e_ff,gamma_ff,epsilon_p):        #diff_eqns_eq_1
    z_val=z(t)
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    
    
    f_m_z_gas=(z_igm*m_dot_cg(z_val,m_h0_val))-(y*(1-R)*f_m_star/m_g)+(y_z*f_m_star)-(eta(z_val,m_h0_val,epsilon_p)*y*f_m_star/m_g)
    return f_m_z_gas

def diff_eqn_eq_zgas_3(t,y,m_g,m_h0_val,e_ff,gamma_ff):       #diff_eqns_eq_2
    z_val=z(t)
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))+(y_z*f_m_star)
    return f_m_z_gas
    
def diff_eqn_eq_zgas_4(t,y,m_g,m_h0_val,e_ff,gamma_ff,epsilon_p):        #diff_eqns_eq_2
    z_val=z(t)
    f_m_star=(e_ff/t_ff(z_val,m_h0_val,gamma_ff))*m_g
    
    f_m_z_gas=(z_igm*m_dot_cg_2(z_val,m_h0_val))-(y*(1-R)*f_m_star/m_g)+(y_z*f_m_star)-(eta(z_val,m_h0_val,epsilon_p)*y*f_m_star/m_g)
    return f_m_z_gas


def diff_eqn_zstar_1(t,y):
    f_m_z_star=0.0
    return f_m_z_star

def diff_eqn_zstar_2(t,y,m_z_g,m_h0_val,e_ff,gamma_ff):
    z_val=z(t)
    f_m_z_star=(m_z_g)*(1-R)*(e_ff/t_ff(z_val,m_h0_val,gamma_ff))
    return f_m_z_star


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
          
        m_dot_star_val=(e_ff/t_ff(redshift,m_h0_val,gamma_ff))*values[0]
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
          
        m_dot_star_val=(e_ff/t_ff(redshift,m_h0_val,gamma_ff))*values[0]
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


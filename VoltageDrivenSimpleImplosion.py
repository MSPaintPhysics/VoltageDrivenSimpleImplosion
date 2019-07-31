"""
Created on Sat Mar 16 18:17:25 2019

@author: Akash & Paul!
"""

import numpy as np
from numpy import pi, log
from scipy import integrate
import matplotlib.pyplot as plt

#Liner Parameters
mu = 4*pi*pow(10,-7)                #magnetic constant (m kg s^-2 A^-2)
C = 0.8e-6                         #Capacitance (Farads)

L_0 = 15e-9                         # Inductance (Henries)
vc = 2*(70*pow(10,3))                     # Voltage across capacitors (Volts)

thickness = 1e-6                                #Liner thickness (m)
r_inner = 0.003175                             #Liner Inner Radius (m)
r_outer = r_inner+thickness                        #Liner Outer Radius (m)
r_rc = 0.019                                # Return Current Radius (m)
de_outer = 2700                       # Density for Al (kg/m**3)
h = 0.020                                   # Height (m)
t_peak = 200E-9                             # Peak rise time (s)
w = pi/(2*t_peak)                           # Frequency (1/s)
Rloss = 2.5                                 # Resistance (Ohms)

m=de_outer*pi*h*(r_outer**2-r_inner**2)     #Mass (kg)

#define indutance equations
def Lvac(r):
    return (mu*h/(2*pi))*log(r_rc/r)
def Lvac_dot(r,v):
    return -(mu*h/(2*pi))*v/r

def f(t,y):
    r,v,I,phi = y
    
    dydt = np.zeros((2,1))
    
    dydt[0] = v #makes the position equation 1st order instead of 2nd order
    dydt[1] = -mu*h*I**2/(4*pi*m*r) #equation for position
    
    dIdt = (phi-I*(Lvac_dot(r,v)))/(L_0+ Lvac(r)) #equation for current
    
    dphidt = (-I-phi/Rloss)/C #equation for voltage
    
    return[dydt[0], dydt[1], dIdt, dphidt]
    
sol = integrate.ode(f).set_integrator('dopri5', method='bdf')

#set timestep size and length of time the ode solver will run
t_start = 0.0 # This is not the greatest variable in the world, no this is just a tribute
t_final = 3*t_peak
delta_t = 1E-10
num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)



#set initial conditions
i_0 = 0
v_0 = 0
sol.set_initial_value([r_outer, v_0, i_0, vc])

t = [t_start]
i = [i_0]
r_out = [r_outer]
v_out = [v_0]
voltage = [vc]

k = 1

while sol.successful() and k < num_steps:
    sol.integrate(sol.t + delta_t)
    
    t.append(sol.t)
    r_out.append(sol.y[0])
    v_out.append(sol.y[1])
    i.append(sol.y[2])
    voltage.append(sol.y[3])
    
    k += 1
    
#scale values
ts = [x*1e9 for x in t]
rs = [x*1e3 for x in r_out]
i_s = [x/1e6 for x in i]
voltage_s = [x/1e6 for x in voltage]    
    
#plot the results    
plt.figure(figsize=(15,15))
plt.plot(ts,rs,linewidth = 4)
plt.xlabel('Time (ns)')
plt.ylabel('Radius (mm)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,15))
plt.plot(ts,v_out,linewidth = 4)
plt.xlabel('Time (ns)')
plt.ylabel('Velocity (m/s)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,15))
plt.plot(ts,i_s,linewidth = 4)
plt.xlabel('Time (ns)')
plt.ylabel('Current (MA)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,15))
plt.plot(ts,voltage_s,linewidth = 4)
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (MV)')
plt.tight_layout()
plt.show()
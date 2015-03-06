# Import the necessary Python modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import minimize

''' This python file hides the programming from the user to create an Ipython
notebook that is for the non-programmer.
created by J. E. Nyquist, March 2015
'''

# Forward Model
import numpy as np
def thiem(Tfac, Ttot, del_h, Ro, Dw):
    '''This function implements equation 2 and 3 
    based on the Thiem Equation'''
    # Convert all quantities to mks
    del_h = del_h * 0.3048 # Convert ft to m
    Ro = float(Ro)*0.3048
    Rw = float(Dw)/2 # Convert well diameter to radius
    Rw = float(Rw)*(0.3048/12) # Convert inches to m
    Ttot = Ttot*0.3048*0.3048/(3600*24) # Convert to m2/sec
    Q = 2.0*np.pi*Tfac*Ttot*del_h/log(float(Ro)/Rw)
    Q = Q * 1000.0 # Convert cubic meters/sec to liters/sec
    Q = Q * 15.8503230745 # Convert L/s to gpm
    return Q
    
# Code to calculate the cumulative flow
def calculate_flow(Tfac,Ttot,del_h,Ro,Dw,ambient=True):
    # Loop over the number of fractures
    Q_sim = np.zeros(len(Tfac))
    Qfrac = 0.0
    for i in range(len(Tfac)):
        if ambient:
            Qfrac += thiem(Tfac[i],Ttot,del_h[i],Ro,Dw)
        else:
            Qfrac += thiem(Tfac[i],Ttot,del_h[i]+drawdown,Ro,Dw)
        Q_sim[i] = Qfrac
    Q_sim = np.flipud(Q_sim) #Reverse to go from top of well down
    return Q_sim
    
# Calculated Flows for given model parameters
Qamb_sim = calculate_flow(Tfac,Ttot,del_h,Ro,Dw,ambient=True)
print('Ambient Flow:', Qamb_sim)
# Pumped flow
Qstress_sim = calculate_flow(Tfac,Ttot,del_h,Ro,Dw,ambient=False)
print('Pumped Flow:', Qstress_sim)

# Interpolate flow values
z = np.arange(0,bot_well,.1)
qa_calc = np.zeros(len(z))
qs_calc = np.zeros(len(z))
qafit_calc = np.zeros(len(z))
qsfit_calc = np.zeros(len(z))
d = [0, bot_well]
d[1:1] = depth
for i in range(len(depth)):
    qa_calc[np.logical_and(z >= d[i], z < d[i+1])] = Qamb_sim[i]
    qs_calc[np.logical_and(z >= d[i], z < d[i+1])] = Qstress_sim[i]
    qafit_calc[np.logical_and(z >= d[i], z < d[i+1])] = Qamb_fit[i]
    qsfit_calc[np.logical_and(z >= d[i], z < d[i+1])] = Qstress_fit[i]
    
    
    # Plot the results
# Axis limits can be adjusted below with xlim and ylim
fig = plt.figure()
fig.set_size_inches(8,10)
fig.subplots_adjust(wspace=0.5)
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim(-0.025,0.025)
ax1.set_ylim(bot_casing,bot_well)
#ax1.xaxis.tick_top()
plt.plot(Qa,Za,'ob')
plt.plot(qa_calc,z,'b--', linewidth=3)
plt.plot(qafit_calc,z,'b-')
plt.title('Ambient Flow')
plt.gca().invert_yaxis()
plt.xlabel('Upward Flow (gpm)')
plt.ylabel('Depth in Feet')
ax2 = fig.add_subplot(1,2,2)
plt.plot(Qs,Zs,'or')
plt.plot(qs_calc,z,'r--', linewidth=3)
plt.plot(qsfit_calc,z,'r-')
plt.title('Pumped Flow')
ax2.set_xlim(-0.5,1.0)
ax2.set_ylim(bot_casing,bot_well)
#ax2.xaxis.tick_top()
plt.gca().invert_yaxis()
plt.xlabel('Upward Flow (gpm)')
plt.ylabel('Depth in Feet')
plt.show()
# Import libraries
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# To-do:
# Ask user for inputs in user input section (once satisfied with script)
# Want to show g's as well as m/s^2 for accel units in chart (2nd y-axis)
# Warn user that this does not take into combustion pressures yet
# Add in BMEP to force calcs
# Estimate peak combustion pressure at say 10-15 deg ATDC instead of BMEP?
# Get consistent with np or not np
# Add half the weight of the conrod
# Add weight of piston rings
# Perhaps take into account the distance of top of piston to wrist pin
# Perhaps expand to take into account bore centerline offset and calculating thrust?
# =============================================================================


# Math functions
pi = np.pi
cos = math.cos
sin = math.sin
sqrt = math.sqrt

# User Inputs ---change this to input() later
NE = 4000  # Engine Speed (rpm)
Stroke = 115.6  # mm
l_conrod = 93.5  # mm
m_piston = 88.63  # grams


# Functions
def time_elapsed(CA):
    return CA*(pi/180)/(NE*2*pi/60)


def piston_pos(CA):  # '\' is a line continuation character
    return r_crank*cos(CA*(pi/180)) + \
           sqrt(l_conrod**2 - r_crank**2*sin(CA*(pi/180))**2)


def piston_vel(CA):
    t = time_elapsed(CA)
    return ((r_crank*omega*sin(omega*t) +
            r_crank**2*omega*sin(2*omega*t) /
            (2*sqrt(l_conrod**2-r_crank**2*sin(omega*t)**2))) *
            (1/1000))


def piston_acc(CA):
    t = time_elapsed(CA)
    return (r_crank**2*omega**2*cos(2*omega*t) /
            sqrt(l_conrod**2-r_crank**2*sin(omega*t)**2) +
            r_crank**4*omega**2*sin(omega*t)*sin(2*omega*t)*cos(omega*t) /
            (2*(l_conrod**2-r_crank**2*sin(omega*t)**2)**(3/2)) +
            r_crank*omega**2*cos(omega*t)) * (1/1000)


# Calculations
CA = np.arange(0, 361, 1)  # crank angle list
r_crank = Stroke/2  # crank throw radius
omega = NE*2*pi*(1/60)  # Engine speed (rad/s)

# Create empty lists
position_piston = []
position_piston_bdc = []  # Will fill this in later, must find bdc first
velocity_piston = []
velocity_piston_abs = []
acceleration_piston = []
acceleration_piston_abs = []
force_piston = []
force_piston_abs = []

for i in CA:  # Fill in empty lists
    pos = piston_pos(i)
    vel = piston_vel(i)
    vel_abs = abs(vel)
    acc = piston_acc(i)
    acc_abs = abs(acc)
    frc = m_piston*acc*(1/1000)*(1/1000)
    frc_abs = abs(frc)

    position_piston.append(pos)
    velocity_piston.append(vel)
    velocity_piston_abs.append(vel_abs)
    acceleration_piston.append(acc)
    acceleration_piston_abs.append(acc_abs)
    force_piston.append(frc)
    force_piston_abs.append(frc_abs)

for i in CA:  # Fill in empty list (needed to find min pos first)
    pos_bdc = piston_pos(i) - min(position_piston)
    position_piston_bdc.append(pos_bdc)

    
# Plot Setup
fig1 = plt.figure(1, figsize=(12, 10))  # Create figure for plots
plt.rcParams.update({'font.size': 10})  # Dictate font size
gs = GridSpec(4, 1, figure=fig1)  # Create a grid to be used to place subplots

# Create the first plot
ax1 = plt.subplot(gs[0, 0])
plt.plot(CA, position_piston,
         color='c', linewidth=1, label='Distance Wrist Pin from Crank Center')
plt.plot(CA, position_piston_bdc,
         '-.', color='c', linewidth=1, label='Position from BDC')
plt.title("Piston Position vs Crank Angle")
plt.xlabel("Crank Angle (deg)")
plt.ylabel("Position (mm)")
plt.grid()
ax1.legend()

ax2 = plt.subplot(gs[1, 0])
plt.plot(CA, velocity_piston, color='r', linewidth=1, label='Velocity')
plt.plot(CA, velocity_piston_abs,
         '--', color='r', linewidth=1, label='Absolute Velocity')
plt.title("Piston Position vs Crank Angle")
plt.xlabel("Crank Angle (deg)")
plt.ylabel("Velocity (m/s)")
plt.grid()
ax2.legend()

ax3 = plt.subplot(gs[2, 0])
plt.plot(CA, acceleration_piston, color='b', linewidth=1, label='Acceleration')
plt.plot(CA, acceleration_piston_abs,
         '--', color='b', linewidth=1, label='Abs Acceleration')
plt.title("Piston Position vs Crank Angle")
plt.xlabel("Crank Angle (deg)")
plt.ylabel("Acceleration (m/s^2)")
plt.grid()
ax3.legend()

ax4 = plt.subplot(gs[3, 0])
plt.plot(CA, force_piston, color='m', linewidth=1, label='Force')
plt.plot(CA, force_piston_abs, '--', color='m', linewidth=1, label='Abs Force')
plt.title("Piston Force vs Crank Angle")
plt.xlabel("Crank Angle (deg)")
plt.ylabel("Force (kN)")
plt.grid()
ax3.legend()

plt.tight_layout()
plt.show()

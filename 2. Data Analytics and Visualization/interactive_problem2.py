import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
import os
import pandas as pd
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
#%matplotlib inline

sim_data = np.loadtxt('https://raw.githubusercontent.com/mhlbapat/python_workshop_series/main/2.%20Data%20Analytics%20and%20Visualization/energy_data.txt')
# print(sim_data)

sim_df = pd.read_csv('https://raw.githubusercontent.com/mhlbapat/python_workshop_series/main/2.%20Data%20Analytics%20and%20Visualization/energy_data.txt', delimiter = ' ')
time = np.array(sim_df['#Timesteps'])
temp_data = sim_df["Temperature"]
cm_temp = np.cumsum(temp_data)/np.arange(1,len(temp_data)+1)
mean_temp = np.ones(np.shape(time))*np.mean(temp_data)
print(sim_df.columns)
PE = sim_df['Potential_Energy']
Total = sim_df['Total_Energy']
press = sim_df['Pressure']

KE = Total - PE
terms = ['Pressure', 'Temperature', 'Potential Energy', 'Total Energy']

fig = plt.figure()
ax = fig.subplots()

ax.bar(terms,[press[2000]*1000, temp_data[2000]*1000, np.abs(PE[2000]), np.abs(Total[2000])])
ax.set_ylim([0,1300])
ax.set_ylabel('Values')
ax_button = plt.axes([0.25, 0.9, 0.65,0.05]) #xposition, yposition, width and height

#Properties of the button
slide_button = Slider(ax_button, 'Timesteps', time[0], time[-1] , 2000) #Slider(ax_button, 'Timesteps',time[1], time[-1])


def update(val):
    fval = np.int(slide_button.val)
    ax.clear()
    ax.set_ylabel('Values')
    ax.bar(terms, [press[fval]*1000, temp_data[fval]*1000, np.abs(PE[fval]), np.abs(Total[2000])])
    ax.set_ylim([0,1300])
    fig.canvas.draw() #redraw the figure





slide_button.on_changed(update)

plt.show()
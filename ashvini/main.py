# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:47:28 2024

@author: Anand Menon
"""

from tabulate import tabulate as table
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy.integrate import solve_ivp

plt.rcParams['figure.dpi'] = 300

H_0=0.0692
e_ff=0.015
A=0.030
omega_m=0.308
omega_b=0.0484
omega_L=0.692
alpha=0.79
pi=math.pi
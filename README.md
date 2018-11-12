# Personal-Programming-Project
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

E = []
A = []
a = []
Error = []

N = 15
n = 12
sigma_0 = 500
sigma_1 = 260
sigma_2 = 280
sigma_00 = 495
sigma_11 = 255
sigma_22 = 275

e = np.arange(0,1,0.038)
    
sigma_exp = sigma_0 + (sigma_1*e) + (sigma_2*(1-np.exp(-N*e)))
sigma_g = sigma_00 + (sigma_11*e) + (sigma_22*(1-np.exp(-n*e)))

error = (1.0/len(e))*((sigma_exp - sigma_g)**2).sum()

E.append(e)
A.append(sigma_exp)
a.append(sigma_g)
Error.append(error)

df = DataFrame({'Plastic Strain': E, 'Stress Experimental': A, 'Stress Guess': a })
df.to_excel('SSData.xlsx', sheet_name='sheet1', index=False)

#print(E)
#print(A)
#print(a)
print(Error)

plt.plot(e, sigma_exp,'R')
plt.plot(e, sigma_g,'G')
plt.show()

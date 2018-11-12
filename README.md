# Personal-Programming-Project-1
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

E = []
A = []

N = 15
sigma_0 = 500
sigma_1 = 260
sigma_2 = 280

e = np.arange(0,1,0.038)
    
sigma_exp = sigma_0 + (sigma_1*e) + (sigma_2*(1-np.exp(-N*e)))

E.append(e)
A.append(sigma_exp)

df = DataFrame({'Plastic Strain': E, 'Stress': A})
df.to_excel('SSData.xlsx', sheet_name='sheet1', index=False)

print(E)
print(A)

plt.plot(e, sigma_exp,'R')
plt.show()

###### imports ######

import numpy as np
from scipy.optimize import minimize

###### Function for reading the experimental values ######

def readData(fname):
    
    f=open(fname,'r')
    lines=f.readlines()
    m=len(lines)
    s=np.zeros(m)
    e=np.zeros(m)
    i=0
    
    for line in lines:
        
      data=line.split()
      e[i]=float(data[0])
      s[i]=float(data[1])
      i+=1
      
    return e,s
 
###### Function for the Voce-Equation ######    

def voce(x,epl):
    
    s0=x[0]  # yield strength
    s1=x[1]  # linear hardening coefficient
    s2=x[2]  # non-linear hardening coefficient
    n=x[3]   # non-linear hardening exponent
    
    sy=s0+s1*epl+s2*(1-np.exp(-n*epl))  # Voce-Equation
    
    return sy

###### Objective function ######

def objfun(x):
    
    global e,s
    m=len(e)
    
    mse=1./m*((voce(x,e)-s)**2).sum() # objective function or cost function
    
    return mse

###### Function for the gradient of the Objective function ######

def grad(x):
    
    global e,s
    m=len(e)
    s2=x[2]
    n=x[3]
    A=np.exp(-n*e)
    B=1-A
    C=voce(x,e)-s
    
    dmseds0=1./m*(2*C).sum()
    dmseds1=1./m*(2*e*C).sum()
    dmseds2=1./m*(2*B*C).sum()
    dmsedn=1./m*(2*s2*e*A*C).sum()
    
    G=np.array([dmseds0,dmseds1,dmseds2,dmsedn]) # gradient matrix
    
    return G

###### Function for the hessian of the Objective function ######

def Hess(x):
    
    global e,s
    m=len(e)
    s2=x[2]
    n=x[3]
    A=np.exp(-n*e)
    A2=np.exp(-2.*n*e)
    B=1-A
    C=voce(x,e)-s
    
    h00=2
    h01=1./m*(2*e).sum()
    h02=1./m*(2*B).sum()
    h03=1./m*(2*s2*e*A).sum()
    h11=1./m*(2*e*e).sum()
    h12=1./m*(2*e*B).sum()
    h13=1./m*(2*s2*e*e*A).sum()
    h22=1./m*(2*B*B).sum()
    h23=1./m*(2*e*A*C+2*s2*e*A*B).sum()
    h33=1./m*(2*s2*s2*e*e*A2-2*s2*e*e*A*C).sum()
    
    H=np.array([                   # hessian matrix
            [h00,h01,h02,h03],
            [h01,h11,h12,h13],
            [h02,h12,h22,h23],
            [h03,h13,h23,h33]])
    
    return H

e,s=readData('epl_sig.txt') # to read data from the file
  
s0=400 # initial guess value for yield strength
s1=200 # initial guess value for linear hardening coefficient
s2=200 # initial guess value for non-linear hardening coefficient
n=20   # initial guess value for non-linear hardening exponent
  
x0=np.array([s0,s1,s2,n]) # array for the starting point

###### Output ######

res = minimize(objfun, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print(res.x)

Res = minimize(objfun, x0, method='trust-ncg', jac=grad, hess=Hess, options={'gtol': 1e-8, 'disp': True})
print(Res.x)
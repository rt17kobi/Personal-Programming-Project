###### imports ######

import numpy as np
import matplotlib.pyplot as plt        # used for plotting the curves

###### simplex-algorithm ######

def simplex(x0, objfun, alpha=1, beta=0.5, gamma=2, delta=0.5, btol=24.899208, maxiter=10000):
    
    '''   
    
    x0: starting point
    objfun: objective function(cost function)
    alpha: reflection parameter
    beta: contraction parameter
    gamma: expansion parameter
    delta: scaling parameter
    btol: breaking criterion of the best point according to scipy optimize
    maxiter: maximum number of steps
    
    '''
    length = len(x0) # length of the starting point
    
    ###### initial guess ######
    
    X0 = x0 + np.array([0,0,0,0])   # origin
    X1 = x0 + np.array([1,0,0,0])   # initial guess with random values and the starting point
    X2 = x0 + np.array([0,1,0,0])   # for four unknown parameters we have to consider five points(n+1)
    X3 = x0 + np.array([0,0,1,0])
    X4 = x0 + np.array([0,0,0,1])
    
    ###### sorting ######
    
    for i in range(maxiter):
               
        point = np.array([X0,X1,X2,X3,X4])  # collection of all the points in an array  
        
        values = np.array([objfun(X0), objfun(X1), objfun(X2), objfun(X3), objfun(X4)]) # value of objective function at all the points  
        
        si = np.argsort(values) # sorting of all the points from best to the worst
              
        b  = point[si[0]]  # best point
        g  = point[si[1]]  # second best point
        g1 = point[si[2]]  # third best point
        g2 = point[si[3]]  # better than worst point or second worst point
        w  = point[si[4]]  # worst point
        
        ###### centroid ######
        
        mid = (g + g1 + g2 + b) / length  # average of all the points excluding the worst point
                
        ###### reflection step ######
        
        xr = mid + alpha * (mid - w)  # reflection point
        
        if objfun(xr) < objfun(b):  # chceking condition between the reflection point and the best point

            ###### expansion step ######

            xe = xr + gamma * (xr - mid)  # expansion point

            if objfun(xe) < objfun(b):  # chceking condition between the expansion point and the best point
                w = xe
            else:
                w = xr
                
        elif objfun(xr) <= objfun(g2):  # chceking condition between the reflection point and the second worst point          
            w = xr
            
        else:
            if objfun(xr) > objfun(w):  # chceking condition between the reflection point and the worst point
                
                ###### inside contraction step ######

                xi = mid - beta * (mid - w) # inside contraction point 

                if objfun(xi) < objfun(w):  # chceking condition between the inside contraction point and the worst point
                    w = xi
                else:  
                    
                    # shrinkage step
                    b  = delta * (b + b)
                    g  = delta * (b + g)
                    g1 = delta * (b + g1)
                    g2 = delta * (b + g2)
                    w  = delta * (b + w)
            else:
                
                ###### outside contraction step ######
                
                xo = mid + beta * (mid - w)  # outside contraction point

                if objfun(xo) <= objfun(xr):  # chceking condition between the outside contraction point and the reflection point
                    w = xo
                else:
                    
                    # shrinkage step
                    b  = delta * (b + b)
                    g  = delta * (b + g)
                    g1 = delta * (b + g1)
                    g2 = delta * (b + g2)
                    w  = delta * (b + w)

        ###### update points ######
        ###### update all the points in each iteration till the breaking criterion is reached ######
        
        X0 = b  # update best point
        X1 = g  # update second best point
        X2 = g1 # update third best point
        X3 = g2 # update second worst point
        X4 = w  # update worst point
        
        print("Result of Nelder-Mead Simplex Algorithm:")
        print("Iteration:", i, "Objective-Function:", objfun(b), "Best-Point:",b) # for checking iteration at each step
        
        if objfun(b) < btol:  # chceking condition for the breaking criterion
            break
              
    return b,g,g1,g2

###### Inputs ###### 
    
if __name__ == '__main__':
    
  ###### Function for reading the standard values ######
  
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


  e,s=readData('epl_sig.txt') # to read data from the file
  
  s0=400 # initial guess value for yield strength
  s1=200 # initial guess value for linear hardening coefficient
  s2=200 # initial guess value for non-linear hardening coefficient
  n=20   # initial guess value for non-linear hardening exponent
  
  x0=np.array([s0,s1,s2,n])
  
  ###### Output ######
  
  b,g,g1,g2 = simplex(x0,objfun)
  
  ###### Plot ######
  
  xopt=simplex(x0,objfun)
  sopt=voce(xopt[-1],e)
  plt.plot(e,s,'r.', label='Simulated-Result')
  plt.plot(e,sopt,'b-', label='Standard-Curve')
  plt.legend(loc='lower right')
  plt.xlabel('Plastic-Strain')
  plt.ylabel('Objective-Function')
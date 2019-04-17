###### imports ######

import numpy as np
import numpy.linalg as ln
from math import sqrt
import matplotlib.pyplot as plt

###### Trust-Region Algorithm ######

def trust_region(x0, objfun, grad, hess, eta=0.125, etav=0.9, etas=0.1, ri=2.0, rd=0.5, gtol=1e-6, r_min=1.0, r_max=10.0, maxiter=10000):
    
    '''
    
    x0: starting point
    objfun: objective function(cost function)
    grad: gradient of the objective function
    hess: hessian of the objective function
    eta: parameter for accepting a step
    etav: checking condition for the expansion of the trust region
    etas: checking condition for shrinking of the trust region
    ri: factor of the radius for the expansion of the trust region
    rd: factor of the radius for shrinking of the trust region
    gtol: break criterion
    r_min: initial trust region radius
    r_max: maximum trust region radius
    maxiter: maximum number of steps
    
    '''
    
    a = x0  # contain all the starting points
    radius = r_min  # initial trust region radius
    k = 0  # initial iteration value
    
    while True:
        
        G = grad(a)  # value of the gradient at the starting point
        H = hess(a)  # value of the hessian at the starting point
        B = np.linalg.inv(H)  # inverse of hessian
        
        P = dogleg(radius, G, H, B)
        
        ###### Actual reduction ######
        
        a_r = objfun(a) - objfun(a + P)

        ###### Predicted reduction ######
        
        p_r = -(np.dot(G, P) + 0.5 * np.dot(P, np.dot(H, P)))
              
        rho = a_r / p_r  # ratio between Actual reduction and Predicted reduction
        
        if p_r == 0.0: # chceking condition for the value of the predicted reduction greater than zero
            rho = 1e99
        else:
            rho = a_r / p_r

        norm_P = sqrt(np.dot(P, P))  # norm

        if rho < etas:  # if rho is close to zero or negative, then shrunk the trust region
            radius = rd * norm_P
        else:
            if rho > etav and abs(norm_P - radius) < gtol:  # if rho is close to one and P has reached the boundary of the trust region, then expand the trust region
                radius = min(ri * radius, r_max)
            else:
                radius = radius

        ###### position for the next iteration ######
        
        if rho > eta:
            a = a + P
        else:
            a = a

        ###### Check if the gradient is small enough to stop ######
        
        if ln.norm(G) < gtol:
            break

        ###### Check if we have looked at enough iterations ######
        
        if k >= maxiter:
            break
        k = k + 1
        
        print("Iteration:", k, "a=",a,"P=", P,"R=",radius,"norm_P=",norm_P,"rho=",rho)
        
    return a

###### Dogleg Step ######        
        
def dogleg(radius, G, H, B):
    
    '''
    
    radius: initial trust region radius
    G: value of the gradient at the starting point
    H: value of the hessian at the starting point
    B: inverse of hessian
    F: full step
    U: unconstrained minimiser along the steepest descent direction
    
    '''
    
    # calculation of Newton point, which is the optimum for the quadratic model function
    
    F = -np.dot(B, G)
    norm_F = sqrt(np.dot(F, F))  # norm

    # Test if the full step is within the trust region
    
    if norm_F <= radius:
        
        print("full step")
        return F
    
    # calculation of Cauchy point, which is the predicted optimum along the direction of steepest descent
    
    U = - (np.dot(G, G) / np.dot(G, np.dot(H, G))) * G
    dot_U = np.dot(U, U)
    norm_U = sqrt(dot_U)  # norm
    
    # If the Cauchy point is outside the trust region, then return the point where the path intersects the boundary
    
    if norm_U >= radius:
        
        print("cauchy outside")
        return radius * U / norm_U
    
    '''
    
    Find the solution to the scalar quadratic equation,
    Compute the intersection of the trust region boundary and the line segment connecting the Cauchy and Newton points,
    This requires solving a quadratic equation
    
    '''
    
    F_U = F - U
    dot_F_U = np.dot(F_U, F_U)
    dot_U_F_U = np.dot(U, F_U)
    fact = dot_U_F_U ** 2 - dot_F_U * (dot_U - radius ** 2)
    tau = (-dot_U_F_U + sqrt(fact)) / dot_F_U

    # Decide on which part of the trajectory to take
    
    print("dogleg step")
    return U + tau * F_U

###### Inputs ######    
    
if __name__ == '__main__':
    
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

  def hess(x):
      
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
    
    H=np.array([                  # hessian matrix
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
  
  x0=np.array([s0,s1,s2,n])
  
  ###### Output ######
  
  result = trust_region(x0, objfun, grad, hess)
  print("Result of Trust-Region algorithm is:",objfun(result),result)
  
  ###### Plot ######
  
  xopt=trust_region(x0, objfun, grad, hess)
  sopt=voce(xopt,e)
  plt.plot(e,s,'r.', label='Simulated-Result')
  plt.plot(e,sopt,'b-', label='Standard-Curve')
  plt.legend(loc='lower right')
  plt.xlabel('Plastic-Strain')
  plt.ylabel('Objective-Function')
###### imports ######

import numpy as np
import math
import numpy.linalg as ln
from math import sqrt

###### Branin function ######

def objfun(x): # objective-function
    
    F=((x[1]-(0.129*x[0]**2)+(1.6*x[0])-6)**2+(6.07*math.cos(x[0]))+10)
    
    return F

###### Function for the gradient of the Objective function ######

def grad(x):
    
    dfdx0=2*(1.6-(0.258*x[0]))*((-0.129*x[0]**2)+(1.6*x[0])+x[1]-6)-(6.07*math.sin(x[0]))
    dfdx1= 2*((-0.129*x[0]**2)+(1.6*x[0])+x[1]-6)
    
    G=np.array([dfdx0,dfdx1]) # gradient matrix
    
    return G

###### Function for the hessian of the Objective function ######

def hess(x):
    
    d2fdx0x0=2*(1.6-(0.258*x[0]))**2-(0.516*((-0.129*x[0]**2)+(1.6*x[0])+x[1]-6))-(6.07*math.cos(x[0]))
    d2fdx0x1=2*(1.6-(0.258*x[0]))
    d2fdx1x0=2*(1.6-(0.258*x[0]))
    d2fdx1x1=2
    
    H=np.array([[d2fdx0x0,d2fdx0x1],[d2fdx1x0,d2fdx1x1]]) # hessian matrix
    
    return H

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

###### inputs ######

x=np.array([6,14])
result = trust_region(x, objfun, grad, hess)

###### outputs ######

print("Result of Trust-Region algorithm is:",objfun(result),result)
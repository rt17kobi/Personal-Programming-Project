###### imports ######

import numpy as np

###### Rosenbrook-function ######

def objfun(x):  # objective-function
    
    f=(1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    
    return f

# simplex algorithm

def simplex(x0, objfun, alpha=1, beta=0.5, gamma=2, delta=0.5, btol=4e-31, maxiter=150):
    
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
    
    X0 = x + np.array([0,0]) # origin
    X1 = x + np.array([1,0]) # initial guess with random values and the starting point
    X2 = x + np.array([0,1]) # for four unknown parameters we have to consider five points(n+1)
    
    ###### sorting ######
    
    for i in range(maxiter):
        
        point = np.array([X0,X1,X2]) # collection of all the points in an array  
        
        values = np.array([objfun(X0), objfun(X1), objfun(X2)]) # value of objective function at all the points 
        
        si = np.argsort(values) # sorting of all the points from best to the worst
        
        b  = point[si[0]] # best point
        g  = point[si[1]] # not best nor worst
        w  = point[si[2]] # worst point
        
        ###### centroid ######
        
        mid = (g + b) / length  # average of all the points excluding the worst point
                
        ###### reflection step ######
        
        xr = mid + alpha * (mid - w)  # reflection point
        
        if objfun(xr) < objfun(b):  # chceking condition between the reflection point and the best point

            ###### expansion step ######

            xe = xr + gamma * (xr - mid)  # expansion point

            if objfun(xe) < objfun(b):  # chceking condition between the expansion point and the best point
                w = xe
            else:
                w = xr
                
        elif objfun(xr) <= objfun(g):  # chceking condition between the reflection point and the second worst point          
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
                    w  = delta * (b + w)

        ###### update points ######
        ###### update all the points in each iteration till the breaking criterion is reached ######
        
        X0 = b  # update best point
        X1 = g  # update not best nor worst
        X2 = w  # update worst point
        
        print("Result of Nelder-Mead Simplex Algorithm:")
        print("Iteration:", i, "Objective-Function:", objfun(b), "Best-Point:",b) # for checking iteration at each step
    
        if objfun(b) < btol:  # chceking condition for the breaking criterion
            break    
    
    return b,g

###### Inputs ###### 

x=np.array([-0.605,0.371])

###### Outputs ######

b,g = simplex(x,objfun)
print("Result of Nelder-Mead Simplex algorithm is:")
print("Best point is:",objfun(b),b)
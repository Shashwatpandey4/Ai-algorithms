import numpy as np 



def softmax(x):
    s = np.exp(x - np.max(x)) 
    return s/s.sum(axis=0)

#-------------------------------------------------------------#

def sigmoid(x):
    return 1/(1+np.exp(x))

#-------------------------------------------------------------#

def init_adam(parameters):

    L = len(parameters)
    v={} #will contain the exponentially weighted average of the gradient. 
    s={} #will contain the exponentially weighted average of the squared gradient.

    #initializing v & s
    #input parameters
    #output  v, s

    for l in range(L):

        v["dw" + str(l+1)] = np.zeros(parameters["w" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dw" + str(l+1)] = np.zeros(parameters["w" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

    return v, s

#-------------------------------------------------------------#

def update_param_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1 = 0.9, beta2= 0.999, epsilon = 1e-8):

    L = len(parameters)//2                  # number of layers in the neural network
    v_corrected = {}                        # initializing first moment estimate
    s_corrected = {}                        # initializing second moment estimate 

   #perform update on all parameters
    for l in range(L):

       v["dw"+str(l+1)] = beta1 * v["dw" + str(l+1)] + (1-beta1) * grads["dw"+str(l+1)]
       v["db"+str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads["db"+str(l+1)]    

       v_corrected["dw"+str(l+1)] = v["dw" + str(l+1)] / (1-beta1**t)
       v_corrected["db"+str(l+1)] = v["db" + str(l+1)] / (1-beta1**t)    

       s["dw"+str(l+1)] = beta2 * s["dw" + str(l+1)] + (1-beta2) * grads["dw"+str(l+1)**2]
       s["db"+str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * grads["db"+str(l+1)**2]    

       s_corrected["dw"+str(l+1)] = s["dw" + str(l+1)] / (1-beta2**t)
       s_corrected["db"+str(l+1)] = s["db" + str(l+1)] / (1-beta2**t)     

       parameters["w"+str(l+1)] = parameters["w" + str(l+1)] - learning_rate * v_corrected["dw" + str(l+1)] / np.sqrt(s_corrected["dw"+str(l+1)] + epsilon)
       parameters["b"+str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db"+str(l+1)] + epsilon)
    

    return parameters, v, s

    
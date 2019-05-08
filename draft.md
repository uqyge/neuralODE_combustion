# Neural ODE for combustion modelling

## Neural network for reaction modelling in combustion
Neural network has been used to flamelet modelling.[]
Combined with unsupervised SOM, neural network has been used for chemistry ode integration for fixed time steps.[]
With the rapid growth in deep learning, modern neural network designs have been used to solve combustion chemistry [] with great success. However so far most application of the deep neural network are still focusing on a static neural network, the computational graph of the network are fixed during the inference. In this paper we show that by combining neural network with ODE numerics, the dynamic nerual ODE network can offer great computational saving and flexbility to chemical reaction modeling in combustion.



## Neural ODE
Common neural network is stationary. 
There are similarity between residual network and ODE.
Combine neural network and numerical ODE solution, a dynamic network can be built for simulating time series. We will use such neural ode network for combustion modelling. 

![Neural ODE](img/nueralODE.jpeg)

## chemical reaction ODE in combustion.
combustion process can be represented as a reaction ODE system coupled with thermal dynamics.

$
dcdt=\omega
$

$
pv=nRT
$

Solving the thermal reaction ode system is the most time comusing part of combustion modelling. 

## neural network design and training
### multi branch bottle neck network
A multi branch bottleneck network is selected as the base network

### numerical gradient training
r2 shows that the learning is successful.
![H2](fig/H2_r2.png)
![HO2](fig/HO2_r2.png)
![H2O](fig/H2O_r2.png)

## ODE numerics
### Euler method
Euler method is the simplest ODE solution method.

$
y_{n+1}=y_{n}+dydt*dt
$

In this work we adopt a adaptive Euler method.

### Equivalent 'depth' of the network
In the dynamic ODENet the depth of the network is equivalently the iteration in the adaptive method. When combined with modern adaptive ODE method, it offers the user great flexibility of balancing accuracy and computational cost. 


## ODENet work for H2 ingition 
H2 is burned in air with a give initial temperature. At given interval $dt=1e-7$ the odeNet results are compared with original ODE solution. The ODENet is able to maintain a stable error. 

![H2](fig/1601_H2.png)
![HO2](fig/1601_HO2.png)
![OH](fig/1601_OH.png)

## conclusion
Neural ODE network combines neural network with ODE solution. The end results, the dynamic ODENet offers a great tool for reaction ODE modelling in combustion simulations. Unlike ealier attents to apply neural network in combustion simulation, ODENet doesn't require specific optimization for each species[], or extra clusttering step like SOM[]. In the H2 ingition test case, the ODENet has shown a steady control of error tolerance. Thanks to the morden network stucture and GPU accelation. ODENet offers a significat boost in reaciton modelling efficiency in combustion modelling.  

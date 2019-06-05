# OpenFOAM meeting
## 1. VS code remote
![vs code remote](./img/vscode_ssh_dask.png)

## 2. Dask
![dask](./img/dask.png)
![vs code remote](./img/vscode_ssh_dask.png)
**Dask in action**
![dInAct](./img/embarrassing.gif)

## 3. Neural ODE for chemical reactions
- **MLP**
![mlp](./img/mlp.webp)
- **Deep Neural Networks**
![res34](./img/res34.png)
![inception](./img/inception.png)
- **Neural ODE**
![odeP](./img/odePaper.png)

- **Runge kutta network**
**RK45**

$k_1=hf(x_n,y_n)$

$k_2=hf(x_n+h/2,y_n+k_1/2)$

$k_3=hf(x_n+h/2,y_n+k_2/2)$

$k_4=hf(x_n+h,y_n+k_3)$

$y_{n+1}=y_n+1/6(k_1 + 2k_2+2k_3+k4)$

**RK45Network**
![rk4Model](./img/rk4Model.png)

**H2 auto ignition**
![OH](fig/euler_1401_OH.png)
![OH](fig/rk4_1401_OH.png)

- **explicit or implicit**
**flame expansion**
![rk45](fig/flame_RK45.png)
![bdf](fig/flame_BDF.png)
![ODE Net](fig/flame_ODENet.png)

## Star me on github
![github](img/github.png)
[https://github.com/uqyge/neuralODE_combustion](https://github.com/uqyge/neuralODE_combustion)
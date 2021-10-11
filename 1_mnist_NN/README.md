## 1_mnist_NN

### 784 - 32 - 32 - 10

A simple neural network.

Training : 60000 images, Testing : 10000 images 

This code can get about **5% error rate in 10 epochs.**

---

### [1_mnist_NN.cpp](1_mnist_NN.cpp)

> S1 : **784 to 32**, using **Sigmoid** as activation func
>
> S2 : **32 to 32**, using **Sigmoid** as activation func
>
> S3 : **32 to 10**, using **Sigmoid** as activation func

Error rate : ~ **4.5%** in 10 epochs

Total Training Time : about **2 min** for 10 epochs

Features : **New**

---

### [1_mnist_NN_Softmax.cpp](1_mnist_NN_Softmax.cpp)

> S1 : **784 to 32**, using Sigmoid as activation func
>
> S2 : **32 to 32**, using Sigmoid as activation func
>
> S3 : **32 to 10**, using **Softmax** as activation func

Error rate : ~ 5.1% in 10 epochs

Total Training Time : about **2 min** for 10 epochs

Features : **Normalization** by Softmax function, and Derivatives of it

---

### ~~1_mnist_NN_ReLu.cpp~~ - Not Working Yet...

> S1 : **784 to 32**, using **ReLu** as activation func
>
> S2 : **32 to 32**, using **ReLu** as activation func
>
> S3 : **32 to 10**, using **Softmax** as activation func

Error rate : ~ ?.?% in 10 epochs

Total Training Time : about ? min for 10 epochs

Features : **Faster Calculation**, Better than Sigmoids.

---

### How to use

You can download exe files to run program, or cpp files to read and edit.

To use programs, you should **download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/)**, and put them **in the same place** with exe file.

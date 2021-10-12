### Does CNN make sigmoid broken?

Using CNN on floor C1 makes floor size bigger, 

from 784(28\*28) to **9216(16@24\*24)**.

sum of all results(by multiplying weights), makes some of the results so big,

so the **results by sigmoid goes to 1 or 0.**

Then, Derivatives of sigmoid goes to 0, so gradient descend doesn't work better.

### That's why I used resize, dividing to some scalar here.

~~~c
	for(i=0; i<32; i++){
		a2[i] += b2[i];                 	//add scalar
		a2[i] /= 100;				//"RESIZE"
		a2[i] = 1.0/(1+exp(-a2[i]));		//Sigmoid
	}
...  
	for(i=0; i<32; i++){
		da2[i] *= a2[i]*(1-a2[i]);		//derivative of sigmoid
		da2[i] /= 100;				//"RESIZE"
		b2[i] += da2[i];			//add dervative on scalar 2
	}
~~~

Without this line, the training doesn't work.

---

### Speed problem

Using CNN takes too long, because the number of calculation grows so high.

So I tried to reduce time, by using strides, max pooling layer, etc.

#### [2_mnist_CNN_MaxPooling.cpp](2_mnist_CNN_MaxPooling.cpp)

> Using max pooling directly into first layer **Didn't Work.**
> 
> : Total training time reduced effectively, of course..
> 
> : But accuracy dropped, since max pooling loses some data...
> 
> : Losing quality of original data was so harsh.

#### [2_mnist_CNN_Stride.cpp](2_mnist_CNN_Stride.cpp)

> Using strides 2 on first CNN layer **Worked.**
> 
> : Total training time also reduced effectively.
> 
> : But, maybe the convolutional net is so small to use strides 2...

#### [2_mnist_CNN_9by9.cpp](2_mnist_CNN_9by9.cpp)

> Using 9 by 9 net, strides 2, 3 and 4 paddings **Worked.**
> 
> : Total traning time also reduced a bit.
> 
> : But the accuracy didn't rise...

---

### Next goal was...

**1. to get 99% accuracy somehow.**

: Maybe using more floors of CNN will work better...

**2. to use ReLu or else activation function, to make the traning much faster.**

: Since sigmoid takes a lot of calculation(exponentional function)...

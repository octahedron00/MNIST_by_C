/*
	by Octo Moon
	I Did It Again!!
	2021.10.01
	
	
	layer 0 : 784 (28*28)
	layer 1 : 32 (sigmoid)
	layer 2 : 32 (sigmoid)
	layer 3 : 10 (softmax)
	
	traning with 60,000 images,
	testing with 10,000 images.
	
	Error rate : 5.05% (10)
	             4.60% (30)
	
*/



#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double x[60006][800], xt[10001][800];
int y[60006], yt[10001];
unsigned char buffer[784];

double w01[784][32], w12[32][32], w23[32][10], dw01[784][32], dw12[32][32], dw23[32][10]; 	//weight
double b1[32], b2[32], b3[10], db1[32], db2[32], db3[10]; 					//bias
double z1[32], z2[32], z3[10], dz1[32], dz2[32], dz3[10]; 					//linear sum
double a1[32], a2[32], a3[10], a4[10], da1[32], da2[32], da3[10], da4[10]; 					//the answer every time



int main(){
	int i, j, n, n1, n2, a=28, s, b=0, t, count=0;
	double r = 0.3, d = 0.1;
	
	FILE *image = fopen("train-images.idx3-ubyte","rb");
	FILE *label = fopen("train-labels.idx1-ubyte","rb");
	n1 = 60000;
	FILE *image_test = fopen("t10k-images.idx3-ubyte","rb");
	FILE *label_test = fopen("t10k-labels.idx1-ubyte","rb");
	n2 = 10000;
	FILE *log = fopen("1_mnist_NN_Softmax_CrossEntropy.log", "w");

	if(!(image&&label)){
		printf("File is not selected");
		return 1;
	}
		
	printf("image file header : ");
	for(i=0; i<16; i++){
		s = getc(image_test);
		s = getc(image);
		printf("%03d ",s);
	}
	printf("\n");
	
	printf("label file header : ");
	for(i=0; i<8; i++){
		s = getc(label_test);
		s = getc(label);
		printf("%03d ",s);
	}
	printf("\n");
	
	for(t=0; t<n1; t++){
		s = getc(label);	
		y[t] = s;	
		b = fread(buffer, 1, sizeof(buffer), image);
		if(b!=784){
			printf("\n%d - something is wrong\n",t);
			return 2;
		}
		
		for(i=0; i<a*a; i++){
			x[t][i] = ((double)buffer[i])/255;	
		}
		
		if(t%1000==0){
			printf("\rdata <");
			for(i=0; i<t/1000; i++){
				printf("=");
			}
			for(i=0; i<60-(t/1000); i++){
				printf("-");
			}
			printf(">");
		}
	}	
	printf("\rdata <");
	for(i=0; i<60; i++){
		printf("=");
	}
	printf(">\n");
	fclose(image);
	fclose(label);
	
	for(t=0; t<n2; t++){
		s = getc(label_test);	
		yt[t] = s;	
		b = fread(buffer, 1, sizeof(buffer), image_test);
		if(b!=784){
			printf("\n%d - something is wrong\n",t);
			return 2;
		}
		
		for(i=0; i<a*a; i++){
			xt[t][i] = ((double)buffer[i])/255;	
		}
		
		if(t%1000==0){
			printf("\rtest <");
			for(i=0; i<t/1000; i++){
				printf("=");
			}
			for(i=0; i<10-(t/1000); i++){
				printf("-");
			}
			printf(">");
		}
	}	
	printf("\rtest <");
	for(i=0; i<10; i++){
		printf("=");
	}
	printf(">\n");
	
	fclose(image_test);
	fclose(label_test);
	printf("\n%lld data is completely served, file closed\n",n1+n2);
	fprintf(log, "\n%lld data is completely served, file closed\n",n1+n2);

	for(i=0; i<784; i++){
		for(j=0; j<32; j++){
			w01[i][j] = ((double)rand()-16384)/16384;
		}
	}
	for(i=0; i<32; i++){
		for(j=0; j<32; j++){
			w12[i][j] = ((double)rand()-16384)/16384;
		}
		b1[i] = ((double)rand()-16384)/16384;
		b2[i] = ((double)rand()-16384)/16384;
	}
	for(i=0; i<32; i++){
		for(j=0; j<10; j++){
			w23[i][j] = ((double)rand()-16384)/16384;
		}
	}
	for(i=0; i<10; i++){
		b3[i] = ((double)rand()-16384)/16384;
	}
	
	
	
	//start now!
	
	printf("\nTotal traning time : ");		// Total traning time, cycle
	scanf("%d",&n);
	fprintf(log, "\nTotal traning time : %d",n);
	
	for(int k=1; k<=n; k++){
		int count_total = 0;
		for(int b=0; b<60000; b++){
			for(i=0; i<32; i++){
				z1[i] = 0;
				z2[i] = 0;
				a1[i] = 0;
				a2[i] = 0;
				da1[i] = 0;
				da2[i] = 0;
			}
			for(i=0; i<10; i++){
				z3[i] = 0;
				a3[i] = 0;
				a4[i] = 0;
				da3[i] = 0;
				da4[i] = 0;
			}
			
			for(i=0; i<784; i++){
				for(j=0; j<32; j++){
					z1[j] += x[b][i]*w01[i][j]; 	//one to one multiple
				}
			}
			for(i=0; i<32; i++){
				z1[i] += b1[i];				//add scalar
				a1[i] = 1.0/(1+exp(-z1[i])); 		//Sigmoid
			}
			
			for(i=0; i<32; i++){
				for(j=0; j<32; j++){
					z2[j] += a1[i]*w12[i][j];	//one to one multiple
				}
			}
			for(i=0; i<32; i++){
				z2[i] += b2[i];				//add scalar
				a2[i] = 1.0/(1+exp(-z2[i]));		//Sigmoid
			}
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					z3[j] += a2[i]*w23[i][j];	//one to one multiple
				}
			}
			for(i=0; i<10; i++){
				z3[i] += b3[i];				//add scalar
				z3[i] = exp(z3[i]);			//Softmax
			}
			for(i=0; i<10; i++){
				a3[i] = z3[i]/(z3[0]+z3[1]+z3[2]+z3[3]+z3[4]+z3[5]+z3[6]+z3[7]+z3[8]+z3[9]);
			}
			
			double max_val = a3[0];
			int max_pos = 0;
			for(i=0; i<10; i++){
				if(max_val<a3[i]){
					max_val = a3[i];
					max_pos = i;
				}
			}
			if(y[b]==max_pos){
				count++;				//check correctness
			}
			
			for(i=0; i<10; i++){
				if(y[b]==i){
					da4[i] += (1/a3[i] - 1);
				}
//				da4[i] -= a3[i];			//build derivative 3
			}
			
			for(i=0; i<10; i++){
				da4[i] *= r;				//weight r
			}
			
			for(i=0; i<10; i++){
				da3[i] += da4[i]*a3[i];			//derivatives of softmax
				for(j=0; j<10; j++){
					da3[i] -= a3[i]*a3[j]*da4[j];
				}
			}
			
			for(i=0; i<10; i++){
				b3[i] += da3[i];			//add derivative on scalar 2
			}
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					da2[i] += w23[i][j]*da3[j];	//build derivative 2
					w23[i][j] += a2[i]*da3[j];	//add derivative on weight 2->3
				}
			}
			
			for(i=0; i<32; i++){
				da2[i] *= a2[i]*(1-a2[i]);		//derivative of sigmoid
				b2[i] += da2[i];			//add dervative on scalar 2
			}
			
			for(i=0; i<32; i++){
				for(j=0; j<32; j++){
					da1[i] += w12[i][j]*da2[j];	//build derivative 1
					w12[i][j] += a1[i]*da2[j];	//add derivative on weight 1->2
				}
			}
			
			for(i=0; i<32; i++){
				da1[i] *= a1[i]*(1-a1[i]);		//derivative of sigmoid
				b1[i] += da1[i];			//add deriative on scalar 1
			}
			
			for(i=0; i<784; i++){
				for(j=0; j<32; j++){
					w01[i][j] += x[b][i]*da1[j];	//add derivative on weight 0->1
				}
			}
			
			if(b%10000==9999){
				printf("\n%d, %5d/60000, success : %d/10000\n", k, b+1, count);
				fprintf(log, "\n%d, %5d/60000, success : %d/10000\n", k, b+1, count);
				count_total+=count;
				count = 0;
			}
		}
		
		printf("\n\n%lf\n\n\n",(double)count_total/60000);
		fprintf(log, "\n\n%lf\n\n\n",(double)count_total/60000);
	}	


	// Total test

	int count_total=0;
	for(int b=0; b<10000; b++){
		for(i=0; i<32; i++){
			z1[i] = 0;
			z2[i] = 0;
			a1[i] = 0;
			a2[i] = 0;
			da1[i] = 0;
			da2[i] = 0;
		}
		for(i=0; i<10; i++){
			z3[i] = 0;
			a3[i] = 0;
			a4[i] = 0;
			da3[i] = 0;
			da4[i] = 0;
		}
		
		for(i=0; i<784; i++){
			for(j=0; j<32; j++){
				z1[j] += xt[b][i]*w01[i][j]; 	//one to one multiple
			}
		}
		for(i=0; i<32; i++){
			z1[i] += b1[i];				//add scalar
			a1[i] = 1.0/(1+exp(-z1[i])); 		//Sigmoid
		}
		
		for(i=0; i<32; i++){
			for(j=0; j<32; j++){
				z2[j] += a1[i]*w12[i][j];	//one to one multiple
			}
		}
		for(i=0; i<32; i++){
			z2[i] += b2[i];				//add scalar
			a2[i] = 1.0/(1+exp(-z2[i]));		//Sigmoid
		}
		
		for(i=0; i<32; i++){
			for(j=0; j<10; j++){
				z3[j] += a2[i]*w23[i][j];	//one to one multiple
			}
		}
		for(i=0; i<10; i++){
			z3[i] += b3[i];				//add scalar
			z3[i] = exp(z3[i]);			//Softmax
		}
		for(i=0; i<10; i++){
			a3[i] = z3[i]/(z3[0]+z3[1]+z3[2]+z3[3]+z3[4]+z3[5]+z3[6]+z3[7]+z3[8]+z3[9]);
		}
		
		double max_val = a3[0];
		int max_pos = 0;
		for(i=0; i<10; i++){
			if(max_val<a3[i]){
				max_val = a3[i];
				max_pos = i;
			}
		}
		if(yt[b]==max_pos){
			count_total++;				//check correctness
		}
		
	}
	printf("\nTotal test result : %lf\n\n",(double)count_total/10000);
	printf("Error rate : %lf",(double)1-(double)count_total/10000);
	fprintf(log, "\nTotal test result : %lf\n\n",(double)count_total/10000);
	fprintf(log, "Error rate : %lf",(double)1-(double)count_total/10000);
	
	/*
	for(i=0; i<784; i++){
		for(j=0; j<32; j++){
			printf("%.2lf ",w01[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	
	for(i=0; i<32; i++){
		for(j=0; j<32; j++){
			printf("%.2lf ",w12[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	
	for(i=0; i<32; i++){
		for(j=0; j<10; j++){
			printf("%.2lf ",w23[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	
	for(i=0; i<32; i++){
		printf("%lf ",a1[i]);
	}
	printf("\n");
	for(i=0; i<32; i++){
		printf("%lf ",a2[i]);
	}
	printf("\n");
	for(i=0; i<10; i++){
		printf("%lf ",a3[i]);
	}
	printf("\n");
	*/
	
	fclose(log);
	getchar();		//hold
	return 0;
}

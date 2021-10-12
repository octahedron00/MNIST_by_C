/*
	by Octo Moon
	Test for compression!
	2021.09.19
	
	
	raw layer : 784 (28*28)
	layer 0 : 196 (14*14) (Max comp)
	layer 1 : 16 * 10*10 (CNN 5*5)
	layer 2 : 16
	layer 3 : 10
	
	traning with 60,000 images,
	testing with 10,000 images.
	
	Error rate : about 2.5% in 10 epochs
		
	Copyright?
*/



#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double x[60006][800], xt[10001][800], xc[60006][15][15], xct[60006][15][15];
int y[60006], yt[10001];
unsigned char buffer[784];

double w01[16][5][5], w12[16][10][10][16], w23[16][10];				//weight
double b1[16], b2[16], b3[10]; 							//linear sum
double a1[16][10][10], a2[16], a3[10], da1[16][10][10], da2[16], da3[10]; 	//the answer every time


double def(){
	return ((double)rand()-16384)/16384;
}


int main(){
	int i, i0, i1, i2, i3, j, j0, j1, j2, j3, n, n1, n2, a=28, s, b=0, t, count=0;
	double r = 1;
	
	FILE *image = fopen("train-images.idx3-ubyte","rb");
	FILE *label = fopen("train-labels.idx1-ubyte","rb");
	n1 = 60000;
	FILE *image_test = fopen("t10k-images.idx3-ubyte","rb");
	FILE *label_test = fopen("t10k-labels.idx1-ubyte","rb");
	n2 = 10000;
	FILE *log = fopen("2_mnist_CNN_MaxPooling.log", "w");

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

	for(int k=0; k<60000; k++){
		for(i=0; i<28; i++){
			for(j=0; j<28; j++){
				if(xc[k][i/2][j/2]<x[k][(i*28)+j]){
					xc[k][i/2][j/2] = x[k][(i*28)+j];
				}
			}
		}
	}
	for(int k=0; k<10000; k++){
		for(i=0; i<28; i++){
			for(j=0; j<28; j++){
				if(xct[k][i/2][j/2]<xt[k][(i*28)+j]){
					xct[k][i/2][j/2] = xt[k][(i*28)+j];
				}
			}
		}
	}
	
	

	// Initialize values, using def() : random

	for(i=0; i<16; i++){
		for(j1=0; j1<5; j1++){
			for(j2=0; j2<5; j2++){
				w01[i][j1][j2] = def();
			}
		}
	}
	for(i0=0; i0<16; i0++){
		for(i1=0; i1<10; i1++){
			for(i2=0; i2<10; i2++){
				for(j=0; j<16; j++){
					w12[i0][i1][i2][j] = def();
				}
				b1[i0] = def();
			}
		}
	}
	for(i=0; i<16; i++){
		for(j=0; j<10; j++){
			w23[i][j] = def();
		}
		b2[i] = def();
	}
	for(i=0; i<10; i++){
		b3[i] = def();
	}
	
	
	
	//start now!
	
	printf("\nTotal traning time : ");		// Total traning time, cycle
	scanf("%d",&n);
	fprintf(log, "\nTotal traning time : %d\n",n);
	
	
	for(int k=1; k<=n; k++){
		fprintf(log, "\n");
		int count_total=0;
		for(int b=0; b<60000; b++){
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						a1[i0][i1][i2] = 0;
						da1[i0][i1][i2] = 0;
					}
				}
			}
			for(i=0; i<16; i++){
				a2[i] = 0;
				da2[i] = 0;
			}
			for(i=0; i<10; i++){
				a3[i] = 0;
				da3[i] = 0;
			}
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						for(j1=0; j1<5; j1++){
							for(j2=0; j2<5; j2++){
								a1[i0][i1][i2] += xc[b][i1+j1][i2+j2]*w01[i0][j1][j2];	// 
							}
						}
					}
				}
			}
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						a1[i0][i1][i2] += b1[i0];
						a1[i0][i1][i2] = 1.0/(1+exp(-a1[i0][i1][i2]));
					}
				}
			}
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						for(j=0; j<16; j++){
							a2[j] += a1[i0][i1][i2]*w12[i0][i1][i2][j];				// 16*(24*24) -> 32
						}
					}
				}
			}
			for(i=0; i<16; i++){
				a2[i] += b2[i];				//add scalar
				a2[i] = 1.0/(1+exp(-a2[i]));		//Sigmoid
			}
			
			for(i=0; i<16; i++){
				for(j=0; j<10; j++){
					a3[j] += a2[i]*w23[i][j];	//one to one multiple					// 32 -> 10
				}
			}
			for(i=0; i<10; i++){
				a3[i] += b3[i];				//add scalar
				a3[i] = 1.0/(1+exp(-a3[i]));		//Sigmoid
			}
			
			double max_val = a3[0];
			int max_pos = 0;
			double mul = r;
			for(i=0; i<10; i++){
				if(max_val<a3[i]){
					max_val = a3[i];
					max_pos = i;
				}
			}
			if(y[b]==max_pos){
				count++;				//check correctness
			}
			else{
				mul = 0.7;
			}
			
			for(i=0; i<10; i++){
				if(y[b]==i){
					da3[i] += 1;
				}
				da3[i] -= a3[i];			//build derivative 3
			}
			for(i=0; i<10; i++){
				da3[i] *= mul;				//r is weight
			}
			
			for(i=0; i<10; i++){
				da3[i] *= a3[i]*(1-a3[i]);		//derivative of sigmoid
				b3[i] += da3[i];			//add derivative on scalar 2
			}
			
			for(i=0; i<16; i++){
				for(j=0; j<10; j++){
					da2[i] += w23[i][j]*da3[j];	//build derivative 2
					w23[i][j] += a2[i]*da3[j];	//add derivative on weight 2->3
				}
			}
			
			for(i=0; i<16; i++){
				da2[i] *= a2[i]*(1-a2[i]);		//derivative of sigmoid
				b2[i] += da2[i];			//add dervative on scalar 2
			}
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						for(j=0; j<16; j++){
							da1[i0][i1][i2] += w12[i0][i1][i2][j]*da2[j];
							w12[i0][i1][i2][j] += a1[i0][i1][i2]*da2[j];
						}
					}
				}
			}
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						da1[i0][i1][i2] *= a1[i0][i1][i2]*(1-a1[i0][i1][i2]);
						b1[i0] += da1[i0][i1][i2];
					}
				}
			}
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						for(j1=0; j1<5; j1++){
							for(j2=0; j2<5; j2++){
								w01[i0][j1][j2] += xc[b][i1+j1][i2+j2]*da1[i0][i1][i2];
							}
						}
					}
				}
			}
			/*	
				printf("%d ",y[b]);
				for(i=0; i<10; i++){
					if(a3[i]>0.8) printf("#");
					else if(a3[i]>0.4) printf("i");
					else if(a3[i]>0.1) printf(".");
					else printf(" ");
				}
				printf("\n");
			*/	
				
			if(b%1000==999){
				printf("\n%d, %5d/60000, success : %d/1000\n", k, b+1, count);
				fprintf(log, "\n%d, %5d/60000, success : %d/1000", k, b+1, count);
				count_total+=count;
				count = 0;
			}
		}
		
		// Total test
	
		count_total=0;
		for(int b=0; b<10000; b++){
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						a1[i0][i1][i2] = 0;
						da1[i0][i1][i2] = 0;
					}
				}
			}
			for(i=0; i<16; i++){
				a2[i] = 0;
				da2[i] = 0;
			}
			for(i=0; i<10; i++){
				a3[i] = 0;
				da3[i] = 0;
			}
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						for(j1=0; j1<5; j1++){
							for(j2=0; j2<5; j2++){
								a1[i0][i1][i2] += xc[b][i1+j1][i2+j2]*w01[i0][j1][j2];	// 
							}
						}
					}
				}
			}
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						a1[i0][i1][i2] += b1[i0];
						a1[i0][i1][i2] = 1.0/(1+exp(-a1[i0][i1][i2]));
					}
				}
			}
			
			for(i0=0; i0<16; i0++){
				for(i1=0; i1<10; i1++){
					for(i2=0; i2<10; i2++){
						for(j=0; j<16; j++){
							a2[j] += a1[i0][i1][i2]*w12[i0][i1][i2][j];				// 16*(24*24) -> 32
						}
					}
				}
			}
			for(i=0; i<16; i++){
				a2[i] += b2[i];				//add scalar
				a2[i] = 1.0/(1+exp(-a2[i]));		//Sigmoid
			}
			
			for(i=0; i<16; i++){
				for(j=0; j<10; j++){
					a3[j] += a2[i]*w23[i][j];	//one to one multiple					// 32 -> 10
				}
			}
			for(i=0; i<10; i++){
				a3[i] += b3[i];				//add scalar
				a3[i] = 1.0/(1+exp(-a3[i]));		//Sigmoid
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
				count_total++;				//check correctness
			}
			if(b%1000==999){		
				printf("\n\n%lf\n\n\n",(double)count_total/1000);
			}
			
		}
		printf("\n%d test result : %lf\n\n",k,(double)count_total/10000);
		printf("Error rate : %lf",(double)1-(double)count_total/10000);
		fprintf(log, "\n%d test result : %lf\n\n",k,(double)count_total/10000);
		fprintf(log, "Error rate : %lf",(double)1-(double)count_total/10000);
	
		
	}	

	for(i0=0; i0<16; i0++){
		for(j1=0; j1<5; j1++){
			for(j2=0; j2<5; j2++){
				if(w01[i0][j1][j2]>0.2) fprintf(log, "+ ");
				else if(w01[i0][j1][j2]<-0.2) fprintf(log, "- ");
				else fprintf(log, "  ");
			}
			fprintf(log, "\n");
		}
		fprintf(log, "\n");
		for(j1=0; j1<5; j1++){
			for(j2=0; j2<5; j2++){
				fprintf(log, "%lf ",w01[i0][j1][j2]);
			}
			fprintf(log, "\n");
		}
		fprintf(log, "\n");
	}
	
	fclose(log);
	getchar();		//hold
	return 0;
}

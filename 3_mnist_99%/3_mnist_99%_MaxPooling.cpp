/*
	by Octo Moon
	I Did It Again!!!
	2021.09.20
	
	
	layer 0 : 784 (28*28) -> padding, 35*35
	layer 1 : 8 * 14*14 (CNN 9*9, padding 3 and 4, stride 2)
	layer 2 : 8 * 7*7 (2*2 Max Pooling)
	layer 3 : 32
	layer 4 : 10
	
	traning with 60,000 images,
	testing with 10,000 images.
	
	
	Error rate : 1.47% (10)
		     1.13% (20)
		     0.82% (30)
		     
		
	Copyright?
*/



#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

double x[60006][900], xt[10001][900], xc[60006][36][36], xtc[10001][36][36];
int y[60006], yt[10001];
unsigned char buffer[784];

double w01[8][9][9], w23[8][7][7][32], w34[32][10];		//weight
double b1[8], b2[8][7][7][5], b3[32], b4[10]; 					//bias
double a1[8][14][14], a2[8][7][7], a3[32], a4[10], a5[10];	//the answer every time
double da1[8][14][14], da2[8][7][7], da3[32], da4[10]; 		//the differentials	


double def(){
	return ((double)rand()-16384)/16384;
}


int main(){
	int i, i0, i1, i2, i3, j, j0, j1, j2, j3, n, n1, n2, a=28, s, b=0, t, count=0;
	double r = 1.0;
	
	FILE *image = fopen("train-images.idx3-ubyte","rb");
	FILE *label = fopen("train-labels.idx1-ubyte","rb");
	n1 = 60000;
	FILE *image_test = fopen("t10k-images.idx3-ubyte","rb");
	FILE *label_test = fopen("t10k-labels.idx1-ubyte","rb");
	n2 = 10000;
	FILE *log = fopen("3_mnist_99%_MaxPooling.log", "w");

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


	//reshape
	
	for(b=0; b<60000; b++){
		for(i=0; i<28; i++){
			for(j=0; j<28; j++){
				xc[b][i+3][j+3] = x[b][i*28+j];
			}
		}
	}
	for(b=0; b<10000; b++){
		for(i=0; i<28; i++){
			for(j=0; j<28; j++){
				xtc[b][i+3][j+3] = xt[b][i*28+j];
			}
		}
	}

	

	// Initialize values, using def() : random

	for(i=0; i<8; i++){
		for(j1=0; j1<9; j1++){
			for(j2=0; j2<9; j2++){
				w01[i][j1][j2] = def();
			}
		}
	}
	for(i0=0; i0<8; i0++){
		for(i1=0; i1<7; i1++){
			for(i2=0; i2<7; i2++){
				for(j=0; j<32; j++){
					w23[i0][i1][i2][j] = def();
				}
				b1[i0] = def();
			}
		}
	}
	for(i=0; i<32; i++){
		for(j=0; j<10; j++){
			w34[i][j] = def();
		}
		b3[i] = def();
	}
	for(i=0; i<10; i++){
		b4[i] = def();
	}
	
	
	
	//start now!
	
	printf("\nTotal traning time : ");		// Total traning time, cycle
	scanf("%d",&n);
	fprintf(log, "\nTotal traning time : %d\n",n);
	
	
	for(int k=1; k<=n; k++){
		fprintf(log, "\n");
		int count_total=0;
		for(int b=0; b<60000; b++){
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						a1[i0][i1][i2] = 0;
						da1[i0][i1][i2] = 0;
					}
				}
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						a2[i0][i1][i2] = 0;
						da2[i0][i1][i2] = 0;
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] = 0;
				da3[i] = 0;
			}
			for(i=0; i<10; i++){
				a4[i] = 0;
				a5[i] = 0;
				da4[i] = 0;
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						for(j1=0; j1<9; j1++){
							for(j2=0; j2<9; j2++){
								a1[i0][i1][i2] += xc[b][i1+i1+j1][i2+i2+j2]*w01[i0][j1][j2];
							}
						}
					}
				}
			}
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						a1[i0][i1][i2] += b1[i0];
						a1[i0][i1][i2] = 1.0/(1+exp(-a1[i0][i1][i2]));
					}
				}
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						a2[i0][i1][i2] = a1[i0][i1+i1][i2+i2];
						b2[i0][i1][i2][1] = i1+i1;
						b2[i0][i1][i2][2] = i2+i2;
						for(j1=0; j1<2; j1++){
							for(j2=0; j2<2; j2++){
								if(a2[i0][i1][i2]<a1[i0][i1+i1+j1][i2+i2+j2]){
									a2[i0][i1][i2] = a1[i0][i1+i1+j1][i2+i2+j2];
									b2[i0][i1][i2][1] = i1+i1+j1;
									b2[i0][i1][i2][2] = i2+i2+j2;
								}
							}
						}
					}
				}
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						for(j=0; j<32; j++){
							a3[j] += a2[i0][i1][i2]*w23[i0][i1][i2][j];				
						}
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] += b3[i];
				a3[i] /= 10;				//add scalar
				a3[i] = 1.0/(1+exp(-a3[i]));		//Sigmoid
			}
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					a4[j] += a3[i]*w34[i][j];	//one to one multiple					// 32 -> 10
				}
			}
			for(i=0; i<10; i++){
				a4[i] += b4[i];				//add scalar
				a4[i] = 1.0/(1+exp(-a4[i]));		//Sigmoid
			}
			
			double max_val = a4[0];
			int max_pos = 0;
			for(i=0; i<10; i++){
				if(max_val<a4[i]){
					max_val = a4[i];
					max_pos = i;
				}
			}
			if(y[b]==max_pos){
				count++;				//check correctness
			}
			
			for(i=0; i<10; i++){
				if(y[b]==i){
					da4[i] += 1;
				}
				da4[i] -= a4[i];			//build derivative 3
			}
			for(i=0; i<10; i++){
				da4[i] *= r;				//r is weight
			}
			
			for(i=0; i<10; i++){
				da4[i] *= a4[i]*(1-a4[i]);		//derivative of sigmoid
				b4[i] += da4[i];			//add derivative on scalar 2
			}
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					da3[i] += w34[i][j]*da4[j];	//build derivative 2
					w34[i][j] += a3[i]*da4[j];	//add derivative on weight 2->3
				}
			}
			
			for(i=0; i<32; i++){
				da3[i] *= a3[i]*(1-a3[i]);
				da3[i] /= 10;				//derivative of sigmoid
				b3[i] += da3[i];			//add dervative on scalar 2
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						for(j=0; j<32; j++){
							int p1 = b2[i0][i1][i2][1];
							int p2 = b2[i0][i1][i2][2];
							da1[i0][p1][p2] += w23[i0][i1][i2][j]*da3[j];
							w23[i0][i1][i2][j] += a2[i0][i1][i2]*da3[j];
						}
					}
				}
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						int p1 = b2[i0][i1][i2][1];
						int p2 = b2[i0][i1][i2][2];
						da1[i0][p1][p2] *= a1[i0][p1][p2]*(1-a1[i0][p1][p2]);
						b1[i0] += da1[i0][p1][p2];
					}
				}
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						int p1 = b2[i0][i1][i2][1];
						int p2 = b2[i0][i1][i2][2];
						for(j1=0; j1<9; j1++){
							for(j2=0; j2<9; j2++){
								w01[i0][j1][j2] += xc[b][p1+p1+j1][p2+p2+j2]*da1[i0][p1][p2];
							}
						}
					}
				}
			}	
			
			next:;
			
			if(b%1000==999){
				printf("\n%d, %5d/60000, success : %d/1000\n", k, b+1, count);
				fprintf(log, "\n%d, %5d/60000, success : %d/1000\n", k, b+1, count);
				count_total+=count;
				count = 0;
			}
			if(b%10000==9999){		
				printf("\n%lf\n\n\n",(double)count_total/10000);
				fprintf(log, "\n%lf\n\n\n",(double)count_total/10000);
				count_total = 0;
			}
		}
		
		
		// Total test
		
		count_total=0;
		for(int b=0; b<10000; b++){
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						a1[i0][i1][i2] = 0;
						da1[i0][i1][i2] = 0;
					}
				}
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						a2[i0][i1][i2] = 0;
						da2[i0][i1][i2] = 0;
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] = 0;
				da3[i] = 0;
			}
			for(i=0; i<10; i++){
				a4[i] = 0;
				a5[i] = 0;
				da4[i] = 0;
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						for(j1=0; j1<9; j1++){
							for(j2=0; j2<9; j2++){
								a1[i0][i1][i2] += xc[b][i1+i1+j1][i2+i2+j2]*w01[i0][j1][j2];
							}
						}
					}
				}
			}
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						a1[i0][i1][i2] += b1[i0];
						a1[i0][i1][i2] = 1.0/(1+exp(-a1[i0][i1][i2]));
					}
				}
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						a2[i0][i1][i2] = a1[i0][i1+i1][i2+i2];
						b2[i0][i1][i2][1] = i1+i1;
						b2[i0][i1][i2][2] = i2+i2;
						for(j1=0; j1<2; j1++){
							for(j2=0; j2<2; j2++){
								if(a2[i0][i1][i2]<a1[i0][i1+i1+j1][i2+i2+j2]){
									a2[i0][i1][i2] = a1[i0][i1+i1+j1][i2+i2+j2];
									b2[i0][i1][i2][1] = i1+i1+j1;
									b2[i0][i1][i2][2] = i2+i2+j2;
								}
							}
						}
					}
				}
			}
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						for(j=0; j<32; j++){
							a3[j] += a2[i0][i1][i2]*w23[i0][i1][i2][j];				
						}
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] += b3[i];
				a3[i] /= 10;				//add scalar
				a3[i] = 1.0/(1+exp(-a3[i]));		//Sigmoid
			}
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					a4[j] += a3[i]*w34[i][j];	//one to one multiple					// 32 -> 10
				}
			}
			for(i=0; i<10; i++){
				a4[i] += b4[i];				//add scalar
				a4[i] = 1.0/(1+exp(-a4[i]));		//Sigmoid
			}
			
			double max_val = a4[0];
			int max_pos = 0;
			for(i=0; i<10; i++){
				if(max_val<a4[i]){
					max_val = a4[i];
					max_pos = i;
				}
			}
			if(y[b]==max_pos){
				count_total++;				//check correctness
			}
		}
		printf("%d test result : %lf\n\n",k,(double)count_total/10000);
		printf("Error rate : %lf\n\n\n",(double)1-(double)count_total/10000);
		fprintf(log, "\n%d test result : %lf\n\n",k,(double)count_total/10000);
		fprintf(log, "Error rate : %lf\n\n\n",(double)1-(double)count_total/10000);
		
		
	}	

	
	for(i0=0; i0<8; i0++){
		for(j1=0; j1<9; j1++){
			for(j2=0; j2<9; j2++){
				if(w01[i0][j1][j2]>0.2) fprintf(log, "+ ");
				else if(w01[i0][j1][j2]<-0.2) fprintf(log, "- ");
				else fprintf(log, "  ");
			}
			fprintf(log, "\n");
		}
		fprintf(log, "\n");
		for(j1=0; j1<9; j1++){
			for(j2=0; j2<9; j2++){
				fprintf(log, "%lf ",w01[i0][j1][j2]);
			}
			fprintf(log, "\n");
		}
		fprintf(log, "\n");
	}
	
	fclose(log);
	getchar();
	return 0;
}

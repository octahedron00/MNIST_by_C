/*
	by Octo Moon
	I Did It Again!!!
	2021.11.05
	
	
	layer 0 : 784 (28*28) -> padding, 35*35
	layer 1 : 8 * 14*14 (CNN 9*9, padding 3 and 4, stride 2)
	layer 2 : 8 * 7*7 (2*2 Max Pooling)
	layer 3 : 32
	layer 4 : 10
	
	traning with 60,000 images,
	testing with 10,000 images.
	
	
	Error rate : 1.50% (10)
		     
		
	Copyright?
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>


double x[60006][900], xt[10001][900], xc[60006][36][36], xtc[10001][36][36];
int y[60006], yt[10001];
unsigned char buffer[784];

double w01[8][9][9], w23[8][7][7][32], w34[32][10];		//weight
double b1[8], b2[8][7][7][5], b3[32], b4[10]; 			//bias
double a1[8][14][14], a2[8][7][7], a3[32], a4[10];		//the answer every time
double da1[8][14][14], da2[8][7][7], da3[32], da4[10]; 		//the differentials
double n2[8][7][7], n3[32], n4[10], bn2[8][7][7], bn3[32], bn4[10];
double dn2[8][7][7], dn3[32], dn4[10], dbn2[8][7][7], dbn3[32], dbn4[10];
double av2, av3, av4, dp2, dp3, dp4, sq2, sq3, sq4;		// values to normalize
double bet2, bet3, bet4, gam2, gam3, gam4;			// BN by beta, gamma - BN = gamma*N + beta	


double def(){
	return ((double)rand()-16384)/32768;
}


int main(){
	int i, i0, i1, i2, i3, j, j0, j1, j2, j3, n, N1, N2, a=28, s, b=0, t, count=0;
	double r = 0.05;
	
	FILE *image = fopen("train-images.idx3-ubyte","rb");
	FILE *label = fopen("train-labels.idx1-ubyte","rb");
	N1 = 60000;
	FILE *image_test = fopen("t10k-images.idx3-ubyte","rb");
	FILE *label_test = fopen("t10k-labels.idx1-ubyte","rb");
	N2 = 10000;
	FILE *log = fopen("4_mnist_BN_MPCNN.log", "w");

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
	
	for(t=0; t<N1; t++){
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
	
	for(t=0; t<N2; t++){
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
	printf("\n%lld data is completely served, file closed\n",N1+N2);
	fprintf(log, "\n%lld data is completely served, file closed\n",N1+N2);


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
	bet2 = def();
	bet3 = def();
	bet4 = def();
	gam2 = def();
	gam3 = def();
	gam4 = def();
	
	
	
	//start now!
	
	printf("\nTotal traning time : ");		// Total traning time, cycle
	scanf("%d",&n);
	fprintf(log, "\nTotal traning time : %d\n",n);
	
	
	for(int k=1; k<=n; k++){
		fprintf(log, "\n");
		int count_total=0;
		double error = 0;
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
						n2[i0][i1][i2] = 0;
						bn2[i0][i1][i2] = 0;
						da2[i0][i1][i2] = 0;
						dn2[i0][i1][i2] = 0;
						dbn2[i0][i1][i2] = 0;
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] = 0;
				n3[i] = 0;
				bn3[i] = 0;
				da3[i] = 0;
				dn3[i] = 0;
				dbn3[i] = 0;
			}
			for(i=0; i<10; i++){
				a4[i] = 0;
				n4[i] = 0;
				bn4[i] = 0;
				da4[i] = 0;
				dn4[i] = 0;
				dbn4[i] = 0;
			}
			
			
			// floor 1
			
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
						if(a1[i0][i1][i2]<0) a1[i0][i1][i2] /= 10;
					}
				}
			}
			
			
			// floor 2
			
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
			
			av2 = 0;
			for(i=0; i<8*7*7; i++){
				av2 += a2[i/49][(i%49)/7][i%7];
			}
			av2 /= (8*7*7);
			dp2 = 0;
			for(i=0; i<8*7*7; i++){
				dp2 += (a2[i/49][(i%49)/7][i%7]-av2)*(a2[i/49][(i%49)/7][i%7]-av2);
			}
			dp2 /= (8*7*7);
			sq2 = sqrt(dp2+0.000001);
			for(i=0; i<8*7*7; i++){
				n2[i/49][(i%49)/7][i%7] = (a2[i/49][(i%49)/7][i%7]-av2)/sq2;
				bn2[i/49][(i%49)/7][i%7] = n2[i/49][(i%49)/7][i%7]*gam2 + bet2;
			}
			
			
			// floor 3
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						for(j=0; j<32; j++){
							a3[j] += bn2[i0][i1][i2]*w23[i0][i1][i2][j];				
						}
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] += b3[i];
				if(a3[i]<0) a3[i] /= 10;
			}
			av3 = 0;
			for(i=0; i<32; i++){
				av3 += a3[i];
			}
			av3 /= 32;
			dp3 = 0;
			for(i=0; i<32; i++){
				dp3 += (a3[i]-av3)*(a3[i]-av3);
			}
			dp3 /= 32;					//get average and dispersion
			sq3 = sqrt(dp3+0.000001);
			for(i=0; i<32; i++){
				n3[i] = (a3[i]-av3)/sq3;
				bn3[i] = n3[i]*gam3 + bet3;		//batch normalization
			}
			
			
			// floor 4
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					a4[j] += bn3[i]*w34[i][j];	//one to one multiple					// 32 -> 10
				}
			}
			for(i=0; i<10; i++){
				a4[i] += b4[i];				//add scalar
				if(a4[i]<0) a4[i] /= 10;		//Leak-ReLU
			}
			av4 = 0;
			for(i=0; i<10; i++){
				av4 += a4[i];
			}
			av4 /= 10;
			dp4 = 0;
			for(i=0; i<10; i++){
				dp4 += (a4[i]-av4)*(a4[i]-av4);
			}
			dp4 /= 10;					//get average and dispersion
			sq4 = sqrt(dp4+0.000001);
			for(i=0; i<10; i++){
				n4[i] = (a4[i]-av4)/sq4;
				bn4[i] = n4[i]*gam4 + bet4;		//batch normalization
			}
			
			
			
			double max_val = bn4[0];
			int max_pos = 0;
			for(i=0; i<10; i++){
				if(max_val<bn4[i]){
					max_val = bn4[i];
					max_pos = i;
				}
			}
			if(y[b]==max_pos){
				count++;				//check correctness
			}
			
			
			
			for(i=0; i<10; i++){
				if(y[b]==i){
					dbn4[i] += 1;
				}
				dbn4[i] -= bn4[i];			//build derivative 3
				error += dbn4[i]*dbn4[i];
			}
			for(i=0; i<10; i++){
				dbn4[i] *= r;				//r is weight
			}
			
			// floor 4
			
			for(i=0; i<10; i++){
				bet4 += dbn4[i];
				gam4 += dbn4[i]*n4[i];			//add derivative on beta and gamma
				dn4[i] = dbn4[i]*gam4;			//derivatives of normalized value
			}
			for(i=0; i<10; i++){
				da4[i] += dn4[i]/sq4;
				for(j=0; j<10; j++){
					da4[i] -= (sq4*sq4 + (a4[i]-av4)*(a4[j]-av4))*dn4[j]/(10*sq4*sq4*sq4);
				}					//derivatives of value after activation function
			}
			for(i=0; i<10; i++){
				if(a4[i]<0) da4[i] /= 10;
				b4[i] += da4[i];			//add derivative on scalar 3
			}
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					dbn3[i] += w34[i][j]*da4[j];	//build derivative 2
					w34[i][j] += bn3[i]*da4[j];	//add derivative on weight 2->3
				}
			}	
			
			
			// floor 3
			
			for(i=0; i<32; i++){
				bet3 += dbn3[i];
				gam3 += dbn3[i]*n3[i];			//add derivative on beta and gamma
				dn3[i] = dbn3[i]*gam3;			//derivatives of normalized value
			}
			for(i=0; i<32; i++){
				da3[i] += dn3[i]/sq3;
				for(j=0; j<32; j++){
					da3[i] -= (sq3*sq3 + (a3[i]-av3)*(a3[j]-av3))*dn3[j]/(32*sq3*sq3*sq3);
				}					//derivatives of value after activation function
			}
			for(i=0; i<32; i++){
				if(a3[i]<0) da3[i] /= 10;
				b3[i] += da3[i];			//add derivative on scalar 3
			}
			for(i=0; i<8*7*7; i++){
				for(j=0; j<32; j++){
					dbn2[i/49][(i%49)/7][i%7] += w23[i/49][(i%49)/7][i%7][j]*da3[j];	//build derivative 2
					w23[i/49][(i%49)/7][i%7][j] += bn2[i/49][(i%49)/7][i%7]*da3[j];		//add derivative on weight 2->3
				}
			}
			
			
			// floor 2
			
			for(i=0; i<8*7*7; i++){
				bet2 += dbn2[i/49][(i%49)/7][i%7];
				gam2 += dbn2[i/49][(i%49)/7][i%7]*n2[i/49][(i%49)/7][i%7];
				dn2[i/49][(i%49)/7][i%7] = dbn2[i/49][(i%49)/7][i%7]*gam2;
				
			}
			for(i=0; i<8*7*7; i++){
				da2[i/49][(i%49)/7][i%7] += dn2[i/49][(i%49)/7][i%7]/sq2;
				for(j=0; j<8*7*7; j++){
					da2[i/49][(i%49)/7][i%7] -= (sq2*sq2 + (a2[i/49][(i%49)/7][i%7]-av2)*(a2[j/49][(j%49)/7][j%7]-av2))*dn2[j/49][(j%49)/7][j%7]/(8*7*7*sq2*sq2*sq2);
				}					//derivatives of value after activation function
			}
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						int p1 = b2[i0][i1][i2][1];
						int p2 = b2[i0][i1][i2][2];
						da1[i0][p1][p2] = da2[i0][i1][i2];
						if(a1[i0][p1][p2]<0) da1[i0][p1][p2] /= 10;
					}
				}
			}
			
			
			// floor 1
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						int p1 = b2[i0][i1][i2][1];
						int p2 = b2[i0][i1][i2][2];
						
						b1[i0] += da1[i0][p1][p2];
						for(j1=0; j1<9; j1++){
							for(j2=0; j2<9; j2++){
								w01[i0][j1][j2] += xc[b][p1+p1+j1][p2+p2+j2]*da1[i0][p1][p2];
							}
						}
					}
				}
			}
			
			if(b%1000==999){
				printf("\n%d, %5d/60000, success : %d/1000\n", k, b+1, count);
				fprintf(log, "\n%d, %5d/60000, success : %d/1000\n", k, b+1, count);
				count_total+=count;
				count = 0;
				printf("Average Loss : %lf\n",error/1000);
				fprintf(log,"Average Loss : %lf\n",error/1000);
				error = 0;
			}
		}
		
		printf("\n%lf\n\n\n",(double)count_total/60000);
		fprintf(log, "\n%lf\n\n\n",(double)count_total/60000);
		count_total = 0;
		
		
		// Total test
		
		count_total=0;
		for(int b=0; b<10000; b++){
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						a1[i0][i1][i2] = 0;
					}
				}
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						a2[i0][i1][i2] = 0;
						n2[i0][i1][i2] = 0;
						bn2[i0][i1][i2] = 0;
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] = 0;
				n3[i] = 0;
				bn3[i] = 0;
			}
			for(i=0; i<10; i++){
				a4[i] = 0;
				n4[i] = 0;
				bn4[i] = 0;
			}
			
			
			// floor 1
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						for(j1=0; j1<9; j1++){
							for(j2=0; j2<9; j2++){
								a1[i0][i1][i2] += xtc[b][i1+i1+j1][i2+i2+j2]*w01[i0][j1][j2];
							}
						}
					}
				}
			}
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<14; i1++){
					for(i2=0; i2<14; i2++){
						a1[i0][i1][i2] += b1[i0];
						if(a1[i0][i1][i2]<0) a1[i0][i1][i2] /= 10;
					}
				}
			}
			
			
			// floor 2
			
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
			
			av2 = 0;
			for(i=0; i<8*7*7; i++){
				av2 += a2[i/49][(i%49)/7][i%7];
			}
			av2 /= (8*7*7);
			dp2 = 0;
			for(i=0; i<8*7*7; i++){
				dp2 += (a2[i/49][(i%49)/7][i%7]-av2)*(a2[i/49][(i%49)/7][i%7]-av2);
			}
			dp2 /= (8*7*7);
			sq2 = sqrt(dp2+0.000001);
			for(i=0; i<8*7*7; i++){
				n2[i/49][(i%49)/7][i%7] = (a2[i/49][(i%49)/7][i%7]-av2)/sq2;
				bn2[i/49][(i%49)/7][i%7] = n2[i/49][(i%49)/7][i%7]*gam2 + bet2;
			}
			
			
			// floor 3
			
			for(i0=0; i0<8; i0++){
				for(i1=0; i1<7; i1++){
					for(i2=0; i2<7; i2++){
						for(j=0; j<32; j++){
							a3[j] += bn2[i0][i1][i2]*w23[i0][i1][i2][j];				
						}
					}
				}
			}
			for(i=0; i<32; i++){
				a3[i] += b3[i];
				if(a3[i]<0) a3[i] /= 10;
			}
			av3 = 0;
			for(i=0; i<32; i++){
				av3 += a3[i];
			}
			av3 /= 32;
			dp3 = 0;
			for(i=0; i<32; i++){
				dp3 += (a3[i]-av3)*(a3[i]-av3);
			}
			dp3 /= 32;					//get average and dispersion
			sq3 = sqrt(dp3+0.000001);
			for(i=0; i<32; i++){
				n3[i] = (a3[i]-av3)/sq3;
				bn3[i] = n3[i]*gam3 + bet3;		//batch normalization
			}
			
			
			// floor 4
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					a4[j] += bn3[i]*w34[i][j];	//one to one multiple					// 32 -> 10
				}
			}
			for(i=0; i<10; i++){
				a4[i] += b4[i];				//add scalar
				if(a4[i]<0) a4[i] /= 10;		//Leak-ReLU
			}
			av4 = 0;
			for(i=0; i<10; i++){
				av4 += a4[i];
			}
			av4 /= 10;
			dp4 = 0;
			for(i=0; i<10; i++){
				dp4 += (a4[i]-av4)*(a4[i]-av4);
			}
			dp4 /= 10;					//get average and dispersion
			sq4 = sqrt(dp4+0.000001);
			for(i=0; i<10; i++){
				n4[i] = (a4[i]-av4)/sq4;
				bn4[i] = n4[i]*gam4 + bet4;		//batch normalization
			}
			
			
			
			double max_val = bn4[0];
			int max_pos = 0;
			for(i=0; i<10; i++){
				if(max_val<bn4[i]){
					max_val = bn4[i];
					max_pos = i;
				}
			}
			if(yt[b]==max_pos){
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

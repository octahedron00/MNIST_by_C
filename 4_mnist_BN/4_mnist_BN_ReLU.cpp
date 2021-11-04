/*
	by Octo Moon
	I Did It!!
	2021.11.05
	
	
	layer 0 : 784 (28*28)
	layer 1 : 32
	layer 2 : 32
	layer 3 : 10
	
	traning with 60,000 images,
	testing with 10,000 images.
	
	Error rate : 3.51% (10)
	
*/



#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double x[60006][800], xt[10001][800];
int y[60006], yt[10001];
unsigned char buffer[784];

double w01[784][32], w12[32][32], w23[32][10], dw01[784][32], dw12[32][32], dw23[32][10]; 	// weight
double b1[32], b2[32], b3[10];									// bias
double z1[32], z2[32], z3[10], dz1[32], dz2[32], dz3[10]; 					// linear sum
double a1[32], a2[32], a3[10], da1[32], da2[32], da3[10]; 					// the answer every time
double n1[32], n2[32], n3[10], bn1[32], bn2[32], bn3[10];					// Normalizations, BNs
double dn1[32], dn2[32], dn3[10], dbn1[32], dbn2[32], dbn3[10];					// derivatives of N, BN (linear sum -> ReLU -> BN by itself, beta, and gamma values)
double av1, av2, av3, dp1, dp2, dp3, sq1, sq2, sq3;						// values to normalize
double bet1, bet2, bet3, gam1, gam2, gam3;							// BN by beta, gamma - BN = gamma*N + beta


double def(){
	return ((double)rand()-16384)/32768;
}

int main(){
	int i, j, n, N1, N2, a=28, s, b=0, t, count=0;
	double r = 0.05, d = 0.1;
	
	FILE *image = fopen("train-images.idx3-ubyte","rb");
	FILE *label = fopen("train-labels.idx1-ubyte","rb");
	N1 = 60000;
	FILE *image_test = fopen("t10k-images.idx3-ubyte","rb");
	FILE *label_test = fopen("t10k-labels.idx1-ubyte","rb");
	N2 = 10000;
	FILE *log = fopen("4_mnist_BN_ReLU.log", "w");

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


	// initialize
	
	for(i=0; i<784; i++){
		for(j=0; j<32; j++){
			w01[i][j] = def();
		}
	}
	for(i=0; i<32; i++){
		for(j=0; j<32; j++){
			w12[i][j] = def();
		}
		b1[i] = def();
		b2[i] = def();
	}
	for(i=0; i<32; i++){
		for(j=0; j<10; j++){
			w23[i][j] = def();
		}
	}
	for(i=0; i<10; i++){
		b3[i] = def();
	}
	bet1 = def();
	bet2 = def();
	bet3 = def();
	gam1 = def();
	gam2 = def();
	gam3 = def();
	
	
	
	//start now!
	
	printf("\nTotal traning time : ");		// Total traning time, cycle
	scanf("%d",&n);
	fprintf(log, "\nTotal traning time : %d",n);
	
	for(int k=1; k<=n; k++){
		int count_total = 0;
		double error = 0;
		for(int b=0; b<60000; b++){
//			printf("\n\n\n");
			double max = -100000000;
			for(i=0; i<32; i++){
				z1[i] = 0;
				z2[i] = 0;
				a1[i] = 0;
				a2[i] = 0;
				n1[i] = 0;
				n2[i] = 0;
				bn1[i] = 0;
				bn2[i] = 0;
				da1[i] = 0;
				da2[i] = 0;
				dn1[i] = 0;
				dn2[i] = 0;
				dbn1[i] = 0;
				dbn2[i] = 0;
			}
			for(i=0; i<10; i++){
				z3[i] = 0;
				a3[i] = 0;
				n3[i] = 0;
				bn3[i] = 0;
				da3[i] = 0;
				dn3[i] = 0;
				dbn3[i] = 0;
			}
			
			// floor 1
			
			for(i=0; i<784; i++){
				for(j=0; j<32; j++){
					z1[j] += x[b][i]*w01[i][j]; 	//one to one multiple
				}
			}
			for(i=0; i<32; i++){
				z1[i] += b1[i];				//add scalar
				a1[i] = z1[i];
				if(a1[i]<0) a1[i] /= 10;		//Leak-ReLu
			}
			av1 = 0;
			for(i=0; i<32; i++){
				av1 += a1[i];
			}
			av1 /= 32;
			dp1 = 0;
			for(i=0; i<32; i++){
				dp1 += (a1[i]-av1)*(a1[i]-av1);
			}
			dp1 /= 32;					//get average and dispersion 
			sq1 = sqrt(dp1+0.000001);
			for(i=0; i<32; i++){
				n1[i] = (a1[i]-av1)/sq1;
				bn1[i] = n1[i]*gam1 + bet1;		//batch normalization
			}
			
//			printf("av1 = %lf, dp1 = %lf, sq1 = %lf\n",av1,dp1,sq1);
//			for(i=0; i<32; i++){
//				printf("%lf ",a1[i]);
//			}
//			printf("\n");
//			for(i=0; i<32; i++){
//				printf("%lf ",n1[i]);
//			}
//			printf("\n%lf*n1 + %lf\n",gam1, bet1);
//			for(i=0; i<32; i++){
//				printf("%lf ",bn1[i]);
//			}
//			printf("\n\n");
			
			
			// floor 2
			
			for(i=0; i<32; i++){
				for(j=0; j<32; j++){
					z2[j] += bn1[i]*w12[i][j];	//one to one multiple
				}
			}
			for(i=0; i<32; i++){
				z2[i] += b2[i];				//add scalar
				a2[i] = z2[i];
				if(a2[i]<0) a2[i] /= 10;		//leak-ReLu
			}
			av2 = 0;
			for(i=0; i<32; i++){
				av2 += a2[i];
			}
			av2 /= 32;
			dp2 = 0;
			for(i=0; i<32; i++){
				dp2 += (a2[i]-av2)*(a2[i]-av2);
			}
			dp2 /= 32;					//get average and dispersion
			sq2 = sqrt(dp2+0.000001);
			for(i=0; i<32; i++){
				n2[i] = (a2[i]-av2)/sq2;
				bn2[i] = n2[i]*gam2 + bet2;		//batch normalization
			}
			
//			printf("av2 = %lf, dp2 = %lf, sq2 = %lf\n",av2,dp2,sq2);
//			for(i=0; i<32; i++){
//				printf("%lf ",a2[i]);
//			}
//			printf("\n");
//			for(i=0; i<32; i++){
//				printf("%lf ",n2[i]);
//			}
//			printf("\n%lf*n2 + %lf\n",gam2, bet2);
//			for(i=0; i<32; i++){
//				printf("%lf ",bn2[i]);
//			}
//			printf("\n\n");
			
			
			// floor 3
			
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					z3[j] += bn2[i]*w23[i][j];	//one to one multiple
				}
			}
			for(i=0; i<10; i++){
				z3[i] += b3[i];				//add scalar
				a3[i] = z3[i];
				if(a3[i]<0) a3[i] /= 10;		//ReLu
			}
			av3 = 0;
			for(i=0; i<10; i++){
				av3 += a3[i];
			}
			av3 /= 10;
			dp3 = 0;
			for(i=0; i<10; i++){
				dp3 += (a3[i]-av3)*(a3[i]-av3);
			}
			dp3 /= 10;					//get average and dispersion
			sq3 = sqrt(dp3+0.000001);
			for(i=0; i<10; i++){
				n3[i] = (a3[i]-av3)/sq3;
				bn3[i] = n3[i]*gam3 + bet3;		//batch normalization
			}
			
//			printf("av3 = %lf, dp3 = %lf, sq3 = %lf\n",av3,dp3,sq3);
//			for(i=0; i<10; i++){
//				printf("%lf ",a3[i]);
//			}
//			printf("\n");
//			for(i=0; i<10; i++){
//				printf("%lf ",n3[i]);
//			}
//			printf("\n%lf*n2 + %lf\n",gam3, bet3);
//			for(i=0; i<10; i++){
//				printf("%lf ",bn3[i]);
//			}
//			printf("\n\n");	
			
			
			// check correctness
			
			double max_val = bn3[0];
			int max_pos = 0;
			for(i=0; i<10; i++){
				if(max_val<bn3[i]){
					max_val = bn3[i];
					max_pos = i;
				}
			}
			if(y[b]==max_pos){
				count++;				//check correctness
			}
			
			
			// initial derivative of dbn3
			
			for(i=0; i<10; i++){
				if(y[b]==i){
					dbn3[i] += 1;			//make values to 0 and 1, just by using batch normalization
				}
				dbn3[i] -= bn3[i];			//build derivative 3
				error += dbn3[i]*dbn3[i];		//error(loss)
			}
			for(i=0; i<10; i++){
				dbn3[i] *= r;				//r is weight	
			}
			
			
			// gradient descend of floor 3
			
			for(i=0; i<10; i++){
				bet3 += dbn3[i];
				gam3 += dbn3[i]*n3[i];			//add derivative on beta and gamma
				dn3[i] = dbn3[i]*gam3;			//derivatives of normalized value
			}
			for(i=0; i<10; i++){
				da3[i] += dn3[i]/sq3;
				for(j=0; j<10; j++){
					da3[i] -= (sq3*sq3 + (a3[i]-av3)*(a3[j]-av3))*dn3[j]/(10*sq3*sq3*sq3);
				}					//derivatives of value after activation function
			}
			for(i=0; i<10; i++){
				if(z3[i]<0) da3[i] /= 10;
				b3[i] += da3[i];			//add derivative on scalar 3
			}
			for(i=0; i<32; i++){
				for(j=0; j<10; j++){
					dbn2[i] += w23[i][j]*da3[j];	//build derivative 2
					w23[i][j] += bn2[i]*da3[j];	//add derivative on weight 2->3
				}
			}
			
			
			// gradient descend of floor 2
			
			for(i=0; i<32; i++){
				bet2 += dbn2[i];
				gam2 += dbn2[i]*n2[i];			//add derivative on beta and gamma
				dn2[i] = dbn2[i]*gam2;			//derivatives of normalized value
			}
			for(i=0; i<32; i++){
				da2[i] += dn2[i]/sq2;
				for(j=0; j<32; j++){
					da2[i] -= (sq2*sq2 + (a2[i]-av2)*(a2[j]-av2))*dn2[j]/(32*sq2*sq2*sq2);
				}					//derivatives of value after activation function
			}
			for(i=0; i<32; i++){
				if(z2[i]<0) da2[i] /= 10;
				b2[i] += da2[i];			//add derivative on scalar 2
			}
			for(i=0; i<32; i++){
				for(j=0; j<32; j++){
					dbn1[i] += w12[i][j]*da2[j];	//build derivative 1
					w12[i][j] += bn1[i]*da2[j];	//add derivative on weight 1->2
				}
			}
			
			
			// gradient descend of floor 1
			
			for(i=0; i<32; i++){
				bet1 += dbn1[i];
				gam1 += dbn1[i]*n1[i];			//add derivative on beta and gamma
				dn1[i] = dbn1[i]*gam1;			//derivatives of normalized value
			}
			for(i=0; i<32; i++){
				da1[i] += dn1[i]/sq1;
				for(j=0; j<32; j++){
					da1[i] -= (sq1*sq1 + (a1[i]-av1)*(a1[j]-av1))*dn1[j]/(32*sq1*sq1*sq1);
				}					//derivatives of value after activation function
			}
			for(i=0; i<32; i++){
				if(z1[i]<0) da1[i] /= 10;
				b1[i] += da1[i];			//add derivative on scalar 1
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
				
				printf("Average Loss : %lf \n", error/10000);
				fprintf(log, "Average Loss : %lf \n", error/10000);
				error = 0;
				
				
//				for(i=0; i<32; i++) printf("%lf ",da3[i]);
//				printf("\n");
//				for(i=0; i<32; i++) printf("%lf ",da2[i]);
//				printf("\n");
//				for(i=0; i<10; i++) printf("%lf ",da1[i]);
//				printf("\n");
			}
		}
		
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
		
		printf("\n\n%lf\n\n\n",(double)count_total/60000);
		fprintf(log, "\n\n%lf\n\n\n",(double)count_total/60000);
	}	


	// Total test

	int count_total=0;
	for(int b=0; b<10000; b++){
		double max = -100000000;
		for(i=0; i<32; i++){
			z1[i] = 0;
			z2[i] = 0;
			a1[i] = 0;
			a2[i] = 0;
			n1[i] = 0;
			n2[i] = 0;
			bn1[i] = 0;
			bn2[i] = 0;
			da1[i] = 0;
			da2[i] = 0;
			dn1[i] = 0;
			dn2[i] = 0;
			dbn1[i] = 0;
			dbn2[i] = 0;
		}
		for(i=0; i<10; i++){
			z3[i] = 0;
			a3[i] = 0;
			n3[i] = 0;
			bn3[i] = 0;
			da3[i] = 0;
			dn3[i] = 0;
			dbn3[i] = 0;
		}
		
		
		// floor 1
		
		for(i=0; i<784; i++){
			for(j=0; j<32; j++){
				z1[j] += xt[b][i]*w01[i][j]; 	//one to one multiple
			}
		}
		for(i=0; i<32; i++){
			z1[i] += b1[i];				//add scalar
			a1[i] = z1[i];
			if(a1[i]<0) a1[i] /= 10;		//ReLu
		}
		av1 = 0;
		for(i=0; i<32; i++){
			av1 += a1[i];
		}
		av1 /= 32;
		dp1 = 0;
		for(i=0; i<32; i++){
			dp1 += (a1[i]-av1)*(a1[i]-av1);
		}
		dp1 /= 32;
		sq1 = sqrt(dp1+0.000001);
		for(i=0; i<32; i++){
			n1[i] = (a1[i]-av1)/sq1;
			bn1[i] = n1[i]*gam1 + bet1;
		}
		
		
		// floor 2
		
		for(i=0; i<32; i++){
			for(j=0; j<32; j++){
				z2[j] += bn1[i]*w12[i][j];	//one to one multiple
			}
		}
		for(i=0; i<32; i++){
			z2[i] += b2[i];				//add scalar
			a2[i] = z2[i];
			if(a2[i]<0) a2[i] /= 10;			//ReLu
		}
		av2 = 0;
		for(i=0; i<32; i++){
			av2 += a2[i];
		}
		av2 /= 32;
		dp2 = 0;
		for(i=0; i<32; i++){
			dp2 += (a2[i]-av2)*(a2[i]-av2);
		}
		dp2 /= 32;
		sq2 = sqrt(dp2+0.000001);
		for(i=0; i<32; i++){
			n2[i] = (a2[i]-av2)/sq2;
			bn2[i] = n2[i]*gam2 + bet2;
		}
		
		
		// floor 3
		
		for(i=0; i<32; i++){
			for(j=0; j<10; j++){
				z3[j] += bn2[i]*w23[i][j];	//one to one multiple
			}
		}
		for(i=0; i<10; i++){
			z3[i] += b3[i];				//add scalar
			a3[i] = z3[i];
			if(a3[i]<0) a3[i] /= 10;		//ReLu
		}
		av3 = 0;
		for(i=0; i<10; i++){
			av3 += a3[i];
		}
		av3 /= 10;
		dp3 = 0;
		for(i=0; i<10; i++){
			dp3 += (a3[i]-av3)*(a3[i]-av3);
		}
		dp3 /= 10;
		sq3 = sqrt(dp3+0.000001);
		for(i=0; i<10; i++){
			n3[i] = (a3[i]-av3)/sq3;
			bn3[i] = n3[i]*gam3 + bet3;
		}
		
		
		// check correctness
		
		double max_val = bn3[0];
		int max_pos = 0;
		for(i=0; i<10; i++){
			if(max_val<bn3[i]){
				max_val = bn3[i];
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
	
	
	fclose(log);
	getchar();		//hold
	return 0;
}

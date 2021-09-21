#include<stdio.h>

double x[60006][800];
int y[60006], z[60006][800];
unsigned char buffer[784];
char s1[10] = {' ', '.', ':', 'l', 'M', '#', '8', '@'};

int main(){
	int i, j, n, a=28, s, b=0, t;
	
	FILE *image = fopen("train-images.idx3-ubyte","rb");
	FILE *label = fopen("train-labels.idx1-ubyte","rb");		
	n = 60000;
//	FILE *image = fopen("t10k-images.idx3-ubyte","rb");			// enable here if you want to read test dataset
//	FILE *label = fopen("t10k-labels.idx1-ubyte","rb");
//	n = 10000;

	if(!(image&&label)){
		printf("File is not selected");
		return 1;
	}
		
	printf("image file header : ");
	for(i=0; i<16; i++){
		s = getc(image);
		printf("%03d ",s);
	}
	printf("\n");
	
	printf("label file header : ");				
	for(i=0; i<8; i++){
		s = getc(label);
		printf("%03d ",s);
	}
	printf("\n");
	
//	fsetpos(image, 16+(28*28*1000));
//	fsetpos(label, 8 +(28*28*1000));

// 	file read
	
	for(t=0; t<60000; t++){
		s = getc(label);	
		y[t] = s;
		
		b = fread(buffer, 1, sizeof(buffer), image);
		if(b!=784){
			printf("\n%d - something is wrong\n");
			return 2;
		}
		
		for(i=0; i<a*a; i++){
			x[t][i] = ((double)buffer[i])/255;	
			z[t][i] = buffer[i]/32;
		}
		
		if(t%1000==0){
			printf("\r<");
			for(i=0; i<t/1000; i++){
				printf("=");
			}
			for(i=0; i<60-(t/1000); i++){
				printf("-");
			}
			printf(">");
		}
	}
	
	printf("\r<");
	for(i=0; i<60; i++){
		printf("=");
	}
	printf(">");

	fclose(image);
	fclose(label);
	
	printf("\n%lld data is completely served, file closed\n",t);
	
//	for(t=59990; t<60000; t++){
//		printf("\ndataset[%d] = %d\n",t,y[t]);
//		for(i=0; i<a; i++){
//			for(j=0; j<a; j++){
//				printf("%.01f ",x[t][(i*a)+j]);
//			}
//			printf("\n");
//		}
//	}
	for(t=59990; t<60000; t++){
		printf("\n\ndataset[%d] = %d + Ascii-Art\n",t,y[t]);			//show dataset, for 59990 ~ 59999 (data : 0 ~ 59999)
		for(i=0; i<a+2; i++){
			printf("--");
		}
		for(i=0; i<a; i++){
			printf("\n| ");
			for(j=0; j<a; j++){
				printf("%c%c",s1[z[t][(i*a)+j]],s1[z[t][(i*a)+j]]);	//show ascii-art of image
			}
			printf(" |");
		}
		printf("\n");
		for(i=0; i<a+2; i++){
			printf("--");
		}
	}
	
	return 0;
}

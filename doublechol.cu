// 14 Oct 2008: Important correction to d_choldc_strip made

// now moving to double precision!!!!!!!

// I am now trying to "transpose" everything and make the code
// return L^T in order to see if the memory related slowdowns go away...

/* (C) Steven Gratton 2008.
   This program computes the Cholesky factorization of a matrix.
   Basic program layout inspired by the examples in the NVIDIA CUDA (TM) SDK.
   Basic algorithm based on that in Numerical Recipes for F90, 
   that referencing Golub and Van Loan, Matrix Computations, Johns
   Hopkins University Press.
   I came up with the parallelization scheme for the G80 myself, but
   (unsurprisingly) it is rather similar to those discussed for general 
   parallel computing, see e.g. SCALAPACK documentation.  I
*/

/* This code is in an unfinished state and as such should not be used
   for anything serious or be redistributed.  

   I plan to make a "proper" version available in the near future.

   Any comments/suggestions are welcome; please see my homepage at
   www.ast.cam.ac.uk/~stg20 for contact details.
*/


// You may need to increase stack size to get the program to run,
// try "ulimit -s unlimited" or similar on linux.


// block indices follow matrix conventions
// i.e. (bp,bq)
// refers to the bp'th block matrix in the bq'th column

// thread x,y indices are always chosen to coalesce global memory reads 


// max block size is 22 because of 512 threads per block limit
// WARNING: various options will now break if this is changed from 16

#define BLOCK_SIZE 16

// MAT_SIZE should be a multiple of BLOCK_SIZE;
// for large matrices you may want to use diaginitialize...

// for BLOCK_SIZE=16, largest matrix you can fit into 8800GTS
// memory is 12496 (i.e. 596 megabytes)
// for a GTX 260 it is 14880=16*930 (=844 MB)
// seems cuda takes just under 50MB in overhead.
// 768, 3072 and 12288 are relevant for healpix...

// 9216=16x576 is too, being 3x3072

// ************************************************************** //
// ******CHANGE THE NUMBER HERE TO TEST PERFORMANCE ISSUE!******* //

#define DESIRED_MAT_SIZE (16*576)

// e.g. for 8800 gts try (16*760)
// for 8800gtx would be interesting to try (16*769) 
// ************************************************************** //


// If DESIRED_MAT_SIZE/16 is a multiple of 20 (on an 8800GTS) consider
// defining GLOBAL_PAD here to speed code back up (by a factor of 2).
// The condition might be a multiple of 24 on an 8800GTX.
// This will slow code down if (DESIRED_MAT_SIZE/16)%20=19 ...

//#define GLOBAL_PAD

#ifdef GLOBAL_PAD
#define MAT_SIZE (DESIRED_MAT_SIZE+16)
#define MAT_BLOCKS (MAT_SIZE/BLOCK_SIZE-1)
#else
#define MAT_SIZE DESIRED_MAT_SIZE
#define MAT_BLOCKS (MAT_SIZE/BLOCK_SIZE)
#endif

// warning: "UNROLL" only set up for a block size of 16!
#define COMPARE
#define NOCPUCMP
#define UNROLL
//#define PRINTMATRIX
//#define FULLINIT
//#define PRINTINPUTMATRIX
//#define MATDISP
//#define CPU

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


//a simple test case...
void diaginitialize(double (*x)[MAT_SIZE])
{
    int i,j;

    for (i=0;i<MAT_SIZE;i++)
    {
	for (j=0;j<MAT_SIZE;j++)
	{
	    x[i][j]=0.;
	}
    }    

    for (j=0;j<MAT_SIZE;j++)
    {
	x[j][j]=j+1.;
    }
}



// generates a random positive definite symmetric matrix
// by adding up the outer products of many (>MAT_SIZE) vectors
void initialize(double (*x)[MAT_SIZE])
{
    double tmp[MAT_SIZE];
    int i,j,a,b;

    for (i=0;i<MAT_SIZE;i++)
    {
	for (j=0;j<MAT_SIZE;j++)
	{
	    x[i][j]=0.;
//	    printf("Zeroing: x[%d][%d]=%f.\n",i,j,x[i][j]);
	}
    
    }

    for (a=0;a<5*MAT_SIZE;a++)
    {
	for (b=0;b<MAT_SIZE;b++)
	{
	    tmp[b]=(1.*(double)rand()) / RAND_MAX;
//	    printf("tmp[%d] for vector %d = %f.\n",b,a,tmp[b]);
	}
	for (i=0;i<MAT_SIZE;i++)
	{
	    for (j=0;j<MAT_SIZE;j++)
	    {
		x[i][j]+=tmp[i]*tmp[j];
//	    printf("Adding: x[%d][%d]+=%f.\n",i,j,x[i][j]);
	    }	
	}
	for (i=0;i<MAT_SIZE;i++)
	{
	    for (j=0;j<MAT_SIZE;j++)
	    {
//	    printf("Final: x[%d][%d]=%f.\n",i,j,x[i][j]);
	    }	
	}
    }
}

void diagonalinitialize(double (*d)[MAT_SIZE],double(*x)[MAT_SIZE])
{
    for (int k=0;k<MAT_SIZE;k++)
    {
	d[0][k]=x[k][k];
    }
}

void cpucholdc(double (*x)[MAT_SIZE],double (*d)[MAT_SIZE])
{
    printf("*Please provide your own cpu implementation...*\n");
}

// this is a small kernel that Cholesky decomposes the current "top left" 
// block of the matrix...

__global__ void d_choldc_topleft(double (*m)[MAT_SIZE],
				 int boffset)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double topleft[BLOCK_SIZE][BLOCK_SIZE+1];

    topleft[ty][tx]=m[ty+BLOCK_SIZE*boffset][tx+BLOCK_SIZE*boffset];

    __syncthreads();

    double diagelem,fac;


// in this loop tx labels column, ty row
    for(int k=0;k<BLOCK_SIZE;k++)
    {
	__syncthreads();
	fac=1./sqrt(topleft[k][k]);
	__syncthreads();
	if ((ty==k)&&(tx>=k)) 
	{
	    topleft[ty][tx]=(topleft[ty][tx])*fac;
	}
	__syncthreads();

	if ((tx>=ty)&&(ty>k)) 
	{
	    topleft[ty][tx]=topleft[ty][tx]-topleft[k][ty]*topleft[k][tx]; 
	}

    }

    __syncthreads();


    if (tx>=ty) {
	m[ty+BLOCK_SIZE*boffset][tx+BLOCK_SIZE*boffset] 
	    =topleft[ty][tx];
    }

}








// this kernel updates the strip below the "topleft" block
__global__ void d_choldc_strip(double (*m)[MAT_SIZE],
			       int blockoffset)
{

// +1 since blockoffset labels the "topleft" position
// and boff is the working position...
    int boffy=blockoffset;
    int boffx = blockIdx.x+blockoffset+1; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    
    __shared__ double topleftt[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ double workingmat[BLOCK_SIZE][BLOCK_SIZE+1];

// deliberately transposed...
    topleftt[tx][ty]=m[ty+blockoffset*BLOCK_SIZE][tx+blockoffset*BLOCK_SIZE];

    workingmat[ty][tx]=
	m[ty+boffy*BLOCK_SIZE][tx+boffx*BLOCK_SIZE];

    __syncthreads();

    // now we forward-substitute for the new strip-elements...
    // one thread per column (a bit inefficient I'm afraid)

    if(ty==0)
    {
	for (int k=0;k<BLOCK_SIZE;k++)
	{
	    double dotprod=0.;
	    for (int m=0;m<k;m++)
	    {
		dotprod+=topleftt[k][m]*workingmat[m][tx];
	    }
	    workingmat[k][tx]=(workingmat[k][tx]-dotprod)/topleftt[k][k];
	}
    }

    __syncthreads();

    m[ty+blockoffset*BLOCK_SIZE][tx+boffx*BLOCK_SIZE]
	=workingmat[ty][tx];
 
}




__global__ void d_choldc_diagupdate(double (*m)[MAT_SIZE],
				    int blockoffset)  
{
    int boffx = blockIdx.x+blockoffset+1; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;

// the +1's stop shared memory bank conflicts when accessing down columns
// There are already no shared bank conflicts when accessing by row

    __shared__ double topt[BLOCK_SIZE][BLOCK_SIZE+1];

// deliberately transposed...
    topt[tx][ty]=m[ty+blockoffset*BLOCK_SIZE][tx+boffx*BLOCK_SIZE];

    __syncthreads();

// ty,tx'th thread works out (ty,tx) cmpt of the product...
    double matrixprod=0.;
     

// C'=C-top^T top = C topt topt^T ...
    if(tx>=ty)  
    {

#ifdef UNROLL

	matrixprod+=topt[ty][0]*topt[tx][0];
	matrixprod+=topt[ty][1]*topt[tx][1];
	matrixprod+=topt[ty][2]*topt[tx][2];
	matrixprod+=topt[ty][3]*topt[tx][3];
	matrixprod+=topt[ty][4]*topt[tx][4];
	matrixprod+=topt[ty][5]*topt[tx][5];
	matrixprod+=topt[ty][6]*topt[tx][6];
	matrixprod+=topt[ty][7]*topt[tx][7];
	matrixprod+=topt[ty][8]*topt[tx][8];
	matrixprod+=topt[ty][9]*topt[tx][9];
	matrixprod+=topt[ty][10]*topt[tx][10];
	matrixprod+=topt[ty][11]*topt[tx][11];
	matrixprod+=topt[ty][12]*topt[tx][12];
	matrixprod+=topt[ty][13]*topt[tx][13];
	matrixprod+=topt[ty][14]*topt[tx][14];
	matrixprod+=topt[ty][15]*topt[tx][15];


#else

	for (int kk=0;kk<BLOCK_SIZE;kk++)
	{
	    matrixprod+=topt[ty][kk]*topt[tx][kk];
	}
    
#endif



	m[ty+boffx*BLOCK_SIZE][tx+boffx*BLOCK_SIZE]-=matrixprod; 
    }
}






// this kernel takes the results of the above ones and applies them to the 
//rest of the matrix...
__global__ void d_choldc_hiupdate(double (*m)[MAT_SIZE],
				  int blockoffset)  
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int boffy=blockIdx.x+blockoffset+1;
    int boffx=boffy+1;

// the +1's stop shared memory bank conflicts when accessing down columns
// There are already no shared bank conflicts when accessing by row

    __shared__ double leftt[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ double rightt[BLOCK_SIZE][BLOCK_SIZE+1];


// now read in the data, always from top right

    int tmpx,tmpy,tmpb;

    tmpy=__mul24(boffy,BLOCK_SIZE);
    tmpb=__mul24(blockoffset,BLOCK_SIZE);

// note the tmpy in the latter term to ensure we get the
// correct common matrix for the row
    leftt[tx][ty]=m[ty+tmpb][tx+tmpy];

    for (;boffx<MAT_BLOCKS;boffx++){


	tmpx=__mul24(boffx,BLOCK_SIZE);



	rightt[tx][ty]=m[ty+tmpb][tx+tmpx];

	__syncthreads();



 
	double matrixprod=0.;

// ty,tx'th thread works out (ty,tx) cmpt of the product...
#ifdef UNROLL


	matrixprod+=leftt[ty][0]*rightt[tx][0];
	matrixprod+=leftt[ty][1]*rightt[tx][1];
	matrixprod+=leftt[ty][2]*rightt[tx][2];
	matrixprod+=leftt[ty][3]*rightt[tx][3];
	matrixprod+=leftt[ty][4]*rightt[tx][4];
	matrixprod+=leftt[ty][5]*rightt[tx][5];
	matrixprod+=leftt[ty][6]*rightt[tx][6];
	matrixprod+=leftt[ty][7]*rightt[tx][7];
	matrixprod+=leftt[ty][8]*rightt[tx][8];
	matrixprod+=leftt[ty][9]*rightt[tx][9];
	matrixprod+=leftt[ty][10]*rightt[tx][10];
	matrixprod+=leftt[ty][11]*rightt[tx][11];
	matrixprod+=leftt[ty][12]*rightt[tx][12];
	matrixprod+=leftt[ty][13]*rightt[tx][13];
	matrixprod+=leftt[ty][14]*rightt[tx][14];
	matrixprod+=leftt[ty][15]*rightt[tx][15];


#else

	for (int kk=0;kk<BLOCK_SIZE;kk++)
	{
	    matrixprod+=leftt[ty][kk]*rightt[tx][kk];
	}
    
#endif

	__syncthreads();

	m[ty+tmpy][tx+tmpx]-=matrixprod;

    }

}



void cpumatdisp(double (*mat)[MAT_SIZE],double (*diag)[MAT_SIZE])
{
    int i,j;
    printf("CPU output: \n");

    for (j=0;j<MAT_SIZE;j++)
    {
	for (i=0;i<MAT_SIZE;i++)
	{
	    printf("%7.4f ",mat[j][i]);
	}
	printf("\n");
    }
    printf("\n");
    for (i=0;i<MAT_SIZE;i++)
    {
	printf("%7.4f ",diag[0][i]);
    }

    printf("\n");
}


void matdisp(double (*matptr)[MAT_SIZE])
{
    double mat[MAT_SIZE][MAT_SIZE];

    unsigned int mat_size=MAT_SIZE*MAT_SIZE*sizeof(double);

    int i,j;
    cudaError_t error;
    cudaThreadSynchronize();

//    printf("In matdisp, matptr=%p.\n\n",matptr);

    cudaMemcpy(mat,matptr, mat_size,cudaMemcpyDeviceToHost);
//	error=cudaGetLastError();
//	printf("In mat disp, Error code %d: %s.\n",error,cudaGetErrorString(error));

    cudaThreadSynchronize();

    printf("\n");

    for (j=0;j<MAT_SIZE;j++)
    {
	for (i=0;i<MAT_SIZE;i++)
	{
	    printf("%7.4f ",mat[j][i]);
	}
	printf("\n");
    }

    printf("\n");


    cudaThreadSynchronize();
}
    


void choldc(double (*mat)[MAT_SIZE])
{
    volatile clock_t gputime;

    double (*d_mat)[MAT_SIZE];
    unsigned int mat_size=MAT_SIZE*MAT_SIZE*sizeof(double);
    cudaMalloc((void**) &d_mat,mat_size);
    cudaMemcpy(d_mat, mat, mat_size,cudaMemcpyHostToDevice);

    cudaError_t error; 


    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 stripgrid;
    dim3 higrid;

    int j=MAT_BLOCKS;
    int i=j;

    gputime=clock();

#ifdef MATDISP
    matdisp(d_mat);
#endif

    while(i>2)
    {
	higrid.x=i-2;
	higrid.y=1;
	higrid.z=1;


	dim3 stripgrid(i-1);


	d_choldc_topleft<<<1,threads>>>(d_mat,j-i);


#ifdef MATDISP
	printf("after topleft for j-i=%d:\n",j-i);
	matdisp(d_mat);
#endif


	d_choldc_strip<<<stripgrid,threads>>>(d_mat,j-i);

#ifdef MATDISP
	matdisp(d_mat);
#endif

	d_choldc_diagupdate<<<stripgrid,threads>>>(d_mat,j-i);

/*
	printf("here,%i %i.\n",higrid.x,higrid.y);

error=cudaGetLastError();
	printf("     Error code %d: %s.\n",error,cudaGetErrorString(error));
*/

#ifdef MATDISP
	matdisp(d_mat);
#endif

//	printf("here,%i %i.\n",higrid.x,higrid.y);

 	d_choldc_hiupdate<<<higrid,threads>>>(d_mat,j-i);

/*
	error=cudaGetLastError();
	printf("     Error code %d: %s.\n",error,cudaGetErrorString(error));
*/

#ifdef MATDISP
	printf("After hiupdate:");
	matdisp(d_mat);
#endif

	i--;
    }

    if(j>1)
    {

	d_choldc_topleft<<<1,threads>>>(d_mat,j-2);

#ifdef MATDISP
	matdisp(d_mat);
#endif

	d_choldc_strip<<<1,threads>>>(d_mat,j-2);

#ifdef MATDISP
	matdisp(d_mat);
#endif

	d_choldc_diagupdate<<<1,threads>>>(d_mat,j-2);

#ifdef MATDISP
	matdisp(d_mat);
#endif
    }

    d_choldc_topleft<<<1,threads>>>(d_mat,j-1);

#ifdef MATDISP
    matdisp(d_mat);
#endif


    cudaThreadSynchronize();
    gputime=clock()-gputime;

    printf("kernel time=%f s.\n",gputime/1.e6f);

    cudaMemcpy(mat,d_mat, mat_size,cudaMemcpyDeviceToHost);
 
    cudaFree(d_mat);
 
    cudaThreadSynchronize();

    error=cudaGetLastError();
    printf("     Error code %d: %s.\n",error,cudaGetErrorString(error));
}


int main()
{
    double x[MAT_SIZE][MAT_SIZE];
    double diagonal[1][MAT_SIZE];
    int i,j;

    clock_t maincputime,maingputime;

    cudaThreadExit();


    printf("Initializing test matrices...\n");
#ifdef FULLINIT
    initialize(x);
#else
    diaginitialize(x);
#endif
    diagonalinitialize(diagonal,x);
#ifdef PRINTINPUTMATRIX
    for (i=0;i<MAT_SIZE;i++)
    {
	for (j=0;j<MAT_SIZE;j++)
	{
	    printf("x[%d][%d]=%f.\n",i,j,x[i][j]);
	}
    }
#endif
#ifdef PRINTINPUTDIAG
    for (i=0;i<MAT_SIZE;i++)
    {
	printf("d[%d]=%f.\n",i,diagonal[0][i]);
    }
#endif

    printf("Cholesky factorizing...\n");

#ifdef COMPARE
    maincputime=clock();
#ifdef NOCPUCMP
#else
    cpucholdc(x,diagonal);
#endif
    maincputime=clock()-maincputime;


#ifdef WARMUP
    printf("gpu warmup...\n");
    maingputime=clock();
    choldc(x);
    maingputime=clock()-maingputime;

    printf("maingputime=%u, maincputime=%u.\n",maingputime,maincputime);

#ifdef FULLINIT
    initialize(x);
#else
    diaginitialize(x);
#endif

    diagonalinitialize(diagonal,x);
#endif

    printf("gpu proper...\n");
    maingputime=clock();
    choldc(x);
    maingputime=clock()-maingputime;

    printf("maingputime=%u, maincputime=%u.\n",maingputime,maincputime);



#else

#ifdef CPU
    cpucholdc(x,diagonal);
#ifdef MATDISP
    cpumatdisp(x,diagonal);
#endif

#else
    choldc(x,diagonal);
#endif
#endif

#ifdef GLOBAL_PAD
    printf("x[%d][%d]=%f.\n",DESIRED_MAT_SIZE-1,DESIRED_MAT_SIZE-1,x[DESIRED_MAT_SIZE-1][DESIRED_MAT_SIZE-1]);
#else
    printf("x[%d][%d]=%f.\n",MAT_SIZE-1,MAT_SIZE-1,x[MAT_SIZE-1][MAT_SIZE-1]);
#endif

#ifdef PRINTMATRIX
    for (j=0;j<MAT_SIZE;j++)
    {
	for (i=0;i<MAT_SIZE;i++)
	{
	    printf("x[%d][%d]=%f.\n",i,j,x[i][j]);
	}
    }
#endif

#ifdef PRINTDIAG
    for (i=0;i<MAT_SIZE;i++)
    {
	printf("d[%d]=%f.\n",i,x[i][i]);
    }
#endif

    return 0;
}

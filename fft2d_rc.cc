// Distributed two-dimensional Discrete FFT transform
// C.V. Chaitanya Krishna
// ECE8893 Project 1

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

//declaring global variables
int numtasks, rank,rc;
int flag2d=0;

Complex **alloc_2d_int(int rows, int cols) 
{
    Complex *data = (Complex *)malloc(rows*cols*sizeof(Complex));
    Complex **array= (Complex **)malloc(rows*sizeof(Complex*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

Complex** StoreIIdata2h(Complex * data, int nrows, int ncols)
{
	Complex** h=alloc_2d_int(nrows,ncols);
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			h[i][j]=*data;
			data++;
		}
	}
	return h;
}

Complex** initW(int nrows, int ncols)
{
	int N=nrows;
	Complex** W=alloc_2d_int(nrows,ncols);
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			W[i][j].real=cos((2*M_PI*i*j)/N);
			W[i][j].imag=-sin((2*M_PI*i*j)/N);
		}
	}
	return W;
}

Complex** initWConj(int nrows, int ncols)
{
	int N=nrows;
	Complex** Wconj=alloc_2d_int(nrows,ncols);
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			Wconj[i][j].real=cos((2*M_PI*i*j)/N);
			Wconj[i][j].imag=sin((2*M_PI*i*j)/N);
		}
	}
	return Wconj;
}

Complex* convArray2DtoArray1D(Complex** resArray2d, int nrows, int ncols)
{
	Complex* resArray1d=new Complex[nrows*ncols];
	int k=0;
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			resArray1d[k]=resArray2d[i][j];
			k++;
		}
	}
	return resArray1d;
}

Complex** Transpose(Complex** res1DFFT, int nrows, int ncols)
{
	Complex** Htran=alloc_2d_int(nrows,ncols);
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			Htran[j][i]=res1DFFT[i][j];
		}
	}
	return Htran;
}

Complex** Transform1D(Complex** W, Complex** h, int nrows, int ncols, int flag2d)
{
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int startrow=(nrows/numtasks)*rank;
  int endrow=startrow+(nrows/numtasks);

  Complex** res1DFFT=alloc_2d_int(nrows,ncols); //2D Array to store final result of 1DFFT
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			res1DFFT[i][j]=0;
		}
	}

  //Allocating memory for 2D Array H to store partial outputs
  Complex** H=alloc_2d_int(nrows,ncols);

  if(rank==0)
  {
  	if(flag2d==0)
  	{
  		cout<<"1D FFT Started\n";
  	}
  }
  
  //Initialize H to 0
  for(int i=0; i<nrows; i++)
  {
  	for(int j=0; j<ncols; j++)
  	{
  		H[i][j]=0;
  	}
  }

  //computing 1D FFT
  for(int i=startrow; i<endrow; i++)
  {
  	for(int j=0; j<ncols; j++)
  	{
  		for(int k=0; k<nrows; k++)
  		{
  			H[i][j]=H[i][j]+(W[j][k]*h[i][k]);
  		}
  	}
  }

  if(rank==0)
  {
  	//Storing output of Rank 0 ONLY
  	for(int i=0; i<nrows; i++)
  	{
  		for(int j=0; j<ncols; j++)
  		{
  			res1DFFT[i][j]=res1DFFT[i][j]+H[i][j];
  		}
  	}

  	for(int tempRank=1; tempRank<numtasks; tempRank++)
  	{
  		//Receive outputs from all Ranks
  		MPI_Status status;
  		rc=MPI_Recv(&(H[0][0]), 2*nrows*ncols, MPI_COMPLEX, tempRank, 0, MPI_COMM_WORLD, &status);

  		if(rc!=MPI_SUCCESS)
  		{
  			cout<<"Receive from Rank "<<rank<<" failed. RC: "<<rc<<endl;
  			MPI_Finalize();
  			exit(1);
  		}
  		//Store outputs from other Ranks in res1DFFT
  		for(int i=0; i<nrows; i++)
  		{
  			for(int j=0; j<ncols; j++)
  			{
  				res1DFFT[i][j]=res1DFFT[i][j]+H[i][j];
  			}
  		}
  		// cout<<"Received from Rank "<<tempRank+rank<<endl;
  	}
  }
  else
  {
  	//Send outputs to Rank 0
  	rc=MPI_Send(&(H[0][0]), 2*nrows*ncols, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
  	//cout<<"Rank "<<rank<<" sending outputs to Rank 0"<<endl;
	if(rc!=MPI_SUCCESS)
  		{
  			cout<<"Send from Rank "<<rank<<" failed. RC: "<<rc<<endl;
  			MPI_Finalize();
  			exit(1);
  		}
  }
  return res1DFFT;

}

Complex** Transform2D(Complex** W, Complex** res1DFFT, int nrows, int ncols, int flag2d) 
{ 
	Complex** res2DFFT=alloc_2d_int(nrows,ncols);	//2D Array to store final result of 2DFFT
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			res2DFFT[i][j]=0;
		}
	}

	//sending 1D FFT results to all ranks
	if(rank==0)
	{	
		if(flag2d==0)
		{
			cout<<"2D FFT Started\n";
			flag2d=1;
		}
		for(int tempRank=1; tempRank<numtasks; tempRank++)
		{	
			rc=MPI_Send(&(res1DFFT[0][0]), 2*nrows*ncols, MPI_COMPLEX, tempRank, 0, MPI_COMM_WORLD);
			if(rc!=MPI_SUCCESS)
			{
				cout<<"Send from rank "<<rank<<" failed"<<endl;
				MPI_Finalize();
				exit(1);
			}
		}
	}
	//receive 1D FFT results from rank 0
	else
	{	
		MPI_Status status;
		rc=MPI_Recv(&(res1DFFT[0][0]), 2*nrows*ncols, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
		if(rc!=MPI_SUCCESS)
		{
			cout<<"Receive from rank "<<rank<<" failed"<<endl;
			MPI_Finalize();
			exit(1);
		}
	}

	Complex** h=Transpose(res1DFFT, nrows, ncols);		//Transposing res1DFFT to perform 1D FFT on columns

	//Perform 1D FFT on columns
	Complex** res2DFFTtran=Transform1D(W, h, nrows, ncols, flag2d);
	res2DFFT=Transpose(res2DFFTtran, nrows, ncols);		//Transposing the results to get it in correct row-column format

	return res2DFFT;
}

Complex** InverseTransform2D(Complex** Wconj, Complex** res2DFFT, int nrows, int ncols)
{
	//Send 2D FFT results to all other ranks
	if(rank==0)
	{
		cout<<"Inverse 2D FFT Started\n";
		for(int tempRank=1; tempRank<numtasks; tempRank++)
		{
			rc=MPI_Send(&(res2DFFT[0][0]), 2*nrows*ncols, MPI_COMPLEX, tempRank, 0, MPI_COMM_WORLD);
			if(rc!=MPI_SUCCESS)
			{
				cout<<"Send from rank "<<rank<<" failed"<<endl;
				MPI_Finalize();
				exit(1);
			}
		}
	}
	//Receive 2D FFT results from rank 0
	else
	{	
		MPI_Status status;
		rc=MPI_Recv(&(res2DFFT[0][0]), 2*nrows*ncols, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
		if(rc!=MPI_SUCCESS)
		{
			cout<<"Receive from rank "<<rank<<" failed"<<endl;
			MPI_Finalize();
			exit(1);
		}
	}

	int flag2d=1;
	Complex** resInv1DFFT=Transform1D(Wconj, res2DFFT, nrows, ncols, flag2d);

	//divide 1D FFT results by N
	int N=ncols;
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			resInv1DFFT[i][j].real=(resInv1DFFT[i][j].real)/N;
			resInv1DFFT[i][j].imag=(resInv1DFFT[i][j].imag)/N;
		}
	}

	Complex** resInv2DFFT=Transform2D(Wconj, resInv1DFFT, nrows, ncols, flag2d);

	//divide 2D FFT results by N
	for(int i=0; i<nrows; i++)
	{
		for(int j=0; j<ncols; j++)
		{
			resInv2DFFT[i][j].real=(resInv2DFFT[i][j].real)/N;
			resInv2DFFT[i][j].imag=(resInv2DFFT[i][j].imag)/N;
		}
	}

	return resInv2DFFT;
}

int main(int argc, char** argv)
{
  rc=MPI_Init(&argc,&argv);
  if(rc!=MPI_SUCCESS)
  		{
  			cout<<"Error starting MPI program. Terminating\n";
  			MPI_Abort(MPI_COMM_WORLD, rc);
  		}

  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  
  InputImage image(fn.c_str());
  int ncols=image.GetWidth();
  int nrows=image.GetHeight();
  Complex * data=image.GetImageData();
 
  Complex** h=StoreIIdata2h(data, nrows, ncols);							//Storing input image data in 2D h matrix		
  Complex** W=initW(nrows, ncols);											//Initialize W matrix
  
  //Perform 1D FFT
  Complex** res1DFFT=Transform1D(W, h, nrows, ncols, flag2d);				//2D array to store results of 1DFFT
  
  //Saving 1D FFT Results
  if(rank==0)
  {
  	Complex* res1DFFT1dArray=convArray2DtoArray1D(res1DFFT, nrows, ncols); 	//1D array to store results of 1DFFT
  	image.SaveImageData("MyAfter1d.txt", res1DFFT1dArray, ncols, nrows);
  	cout<<"1D FFT Complete\n\n";
  }
  
  //Perform 2D FFT
  Complex** res2DFFT=Transform2D(W, res1DFFT, nrows, ncols, flag2d);		//2D array to store results of 2DFFT

  //Saving 2D FFT Results
  if(rank==0)
  {
  	Complex* res2DFFT1dArray=convArray2DtoArray1D(res2DFFT, nrows, ncols);	//1D array to store results of 2DFFT
  	image.SaveImageData("MyAfter2D.txt", res2DFFT1dArray, ncols, nrows);
  	cout<<"2D FFT Complete\n"<<endl;
  }

  
  Complex** Wconj=initWConj(nrows, ncols);									//Initialize W Conjugate matrix

  //Perform Inverse 2D FFT
  Complex** resInvFFT=InverseTransform2D(Wconj, res2DFFT, nrows, ncols);	//2D array to store results of Inverse 2DFFT

  //Saving Inverse 2D FFT Results
  if(rank==0)
  {
  	Complex* resInvFFT1dArray=convArray2DtoArray1D(resInvFFT, nrows, ncols);//1D array to store results of Inverse 2DFFT
  	image.SaveImageData("MyAfterInverse.txt", resInvFFT1dArray, ncols, nrows);
  	cout<<"Inverse 2D FFT Complete\n\n"<<endl;
  }

  MPI_Finalize();
  return 0;
}
#include "Header.h"
#include "cuda_runtime.h"

__device__ char compareCharacters(char c1, char c2);
__global__ void calculateSimilarityChar(char* seq1, char* seq2, char* signs, int length, int offset, int mutationIndex);
__constant__ char conservativeGroups[9][5] = {"NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF"};
__constant__ char semiConservativeGroups[11][7] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};

 
__global__ void calculateSimilarityChar(char* seq1, char* seq2, char* signs, int length, int offset, int mutationIndex) 
{
//	Fill the signs array with a similarity sequence as this: ::*..*.**:: * . :: .

    	int i = blockDim.x * blockIdx.x + threadIdx.x;
	char c1, c2;
    	// compare the characters and write to signs in the corresponding positions accoarding to the thread ID 
    	if (i < length)
	{
		c1 = seq1[i + offset];
		c2 = seq2[i];
		if(i == mutationIndex)
		{
			*(signs + i) = ' ';
			return;
		}
		else if(i > mutationIndex)
			c2 = seq2[i - 1];

		*(signs + i ) = compareCharacters(c1, c2);
	}

}


__device__ char compareCharacters(char c1, char c2)
{	
//	Given two Characters, compute the result Character with regard to the conservative and semi-conservative groups.
	char* s;
	int i, j, containsC1 = 0, containsC2 = 0;
	char c;
	
	
	if(c1 == c2)
		return '*';
	for(i=0; i<9 ;i++)
	{
		s = conservativeGroups[i];
		containsC1 = 0;
		containsC2 = 0;
		for (j=0; j<4; j++) 
		{
			c = s[j];
			if(c == '\0')
				break;
			if(c == c1)
			{
				containsC1 = 1;
				if(containsC2)
					return ':';
			}
			if(c == c2)
			{
				containsC2 = 1;
				if(containsC1)
					return ':';
			}				
		}
		
	}
	
	for(i=0; i<11 ;i++)
	{
		s = semiConservativeGroups[i];		
		containsC1 = 0;
		containsC2 = 0;	
		for (j=0; j<6; j++) 
		{
			c = s[j];
			if(c == '\0')
				break;
			if(c == c1)
			{
				containsC1 = 1;
				if(containsC2)
					return '.';
			}
			if(c == c2)
			{
				containsC2 = 1;
				if(containsC1)
					return '.';
			}				
		}

	}
	return ' ';

}
		

cudaError_t allocateCudaMemory(char** seq1, char** seq2, char** signs, int msLength, int sLength)
{
	// Given String array pointers, allocate them into the CUDA memory.

	// Error code to check return values for CUDA calls
    	cudaError_t err1 = cudaSuccess;
    	cudaError_t err2 = cudaSuccess;
    	cudaError_t err3 = cudaSuccess;


    	size_t size1 = msLength * sizeof(char);
    	size_t size2 = sLength * sizeof(char);
	
	

	
    	// Allocate memory on GPU to copy the data from the host
    	err1 = cudaMalloc(seq1, size1);
    	err2 = cudaMalloc(seq2, size2);
    	err3 = cudaMalloc(signs, size2);
	
	if (err1 != cudaSuccess) 
	{
        	fprintf(stderr, "1Failed to allocate device memory - %s\n", cudaGetErrorString(err1));
        	exit(EXIT_FAILURE);
    	}

	if (err2 != cudaSuccess) 
	{
        	fprintf(stderr, "2Failed to allocate device memory - %s\n", cudaGetErrorString(err2));
        	exit(EXIT_FAILURE);
    	}
	
	if (err3 != cudaSuccess) 
	{
        	fprintf(stderr, "3Failed to allocate device memory - %s\n", cudaGetErrorString(err3));
        	exit(EXIT_FAILURE);
    	}

	return cudaSuccess;

}

cudaError_t copyInformationToCuda(char* seq1, char* seq2, MainSequence* ms, Sequence* s)
{
	// Given empty String pointers after alocation inside CUDA, copy the original information to them for further computation.
	// Error code to check return values for CUDA calls
    	cudaError_t err1 = cudaSuccess;
    	cudaError_t err2 = cudaSuccess;



    	size_t size1 = ms->length * sizeof(char);
    	size_t size2 = s->length * sizeof(char);
	
	
	// Copy data from host to the GPU memory
	err1 = cudaMemcpy(seq1, ms->letters, size1, cudaMemcpyHostToDevice);
	err2 = cudaMemcpy(seq2, s->letters, size2, cudaMemcpyHostToDevice);

	if (err1 != cudaSuccess || err2 != cudaSuccess) 
	{
		fprintf(stderr, "4Failed to copy data from host to device - %s\n", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}
	
	if (err2 != cudaSuccess) 
	{
		fprintf(stderr, "5Failed to copy data from host to device - %s\n", cudaGetErrorString(err2));
		exit(EXIT_FAILURE);
	}

	return cudaSuccess;

}

void freeCudaMemory(char* seq1, char* seq2, char* signs)
{

	// Free allocated memory on GPU
	if (cudaFree(signs) != cudaSuccess || cudaFree(seq1) != cudaSuccess || cudaFree(seq2) != cudaSuccess) 
	{
		fprintf(stderr, "8Failed to free device data");
		exit(EXIT_FAILURE);
	}

}

int GPU_Create_Signs(Sequence* s, int n, char* originalSigns, int mutationIndex, char* seq1, char* seq2, char* signs)
{
	// Given the Alocated and copied String Sequences, use the CUDA kernel to compute the target similarity String according to the given mutation and offset n.
	// Error code to check return values for CUDA calls
    	cudaError_t err = cudaSuccess;
    	size_t size = s->length * sizeof(char);
	
	
	// Launch the Kernel
	int threadsPerBlock = 100;
	int blocksPerGrid = (s->length + threadsPerBlock) / threadsPerBlock; // added + 1 for '-'

	calculateSimilarityChar<<<blocksPerGrid, threadsPerBlock>>>(seq1, seq2, signs, s->length + 1, n, mutationIndex); // including the '-' (this is the + 1)

	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		fprintf(stderr, "6Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the  result from GPU to the host memory.
	err = cudaMemcpy(originalSigns, signs, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) 
	{
		fprintf(stderr, "7Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

		
	
	return 0;
}

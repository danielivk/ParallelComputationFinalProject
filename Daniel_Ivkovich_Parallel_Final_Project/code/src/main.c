#include <mpi.h>
#include <omp.h>
#include "Header.h"

 

int main(int argc, char *argv[]) {
	// MPI variables
	int rank, size;
	MPI_Status  status;


	// Loop variables and scores
	int i, n, k;
	float currentScore, bestScore;
	int mutationIndex, offset, best_offset = -1, bestMutation = -1;
	
	// Time Calculation
	double t1, t2;

	// CUDA variables
	char *signs, *seq1, *seq2;
	char* originalSigns;


	// INPUT from file
	MainSequence ms;
	Sequence** sequences;
	int sequenceCount;
	Sequence* currentSequence;
	FILE* f;
	

	// MPI Setup
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	t1 = MPI_Wtime();

	// All Processes Read the input
	readAllSequences(&ms, &sequences, &sequenceCount);
	f = fopen(OUTPUT_FILE_PATH, "w+"); // file for output

	// for each sequence in the input compute the optimal offset and mutation
	for(i = 0; i < sequenceCount; i++)
	{
		currentSequence = sequences[i];
		offset = ms.length - currentSequence->length + 1;
		originalSigns = (char*) malloc (currentSequence->length * sizeof(char));
		bestScore = - ms.w[3] * 3000; 							// minimum value (if they all miss) for the algorithm iteration reset


		allocateCudaMemory(&seq1, &seq2, &signs, ms.length, currentSequence->length); 	//alocation for the CUDA variables.
		copyInformationToCuda(seq1, seq2, &ms, currentSequence); 			// copy memory of sequences from ms and currentSequence to seq1 and seq2 on CUDA!

		// Every proccess calculates its share of the computations offset wise and includes all combinations for its share.
		// MPI divides the job into proccesses. Each process uses threads with OMP, each thread will call CUDA to compute the similarity String.
		#pragma omp for
		for (mutationIndex = 1; mutationIndex < currentSequence->length; mutationIndex++)
			findBestCombination(rank, size, offset, mutationIndex, currentScore, &bestScore, &best_offset, &bestMutation, currentSequence, originalSigns, seq1, seq2, signs, ms.w);
			
		if(rank == 0) 	
		{

			for(k=1; k < size; k++)
			{			
				// compare best score of proccess 0 with best score of procces k
				MPI_Recv (&currentScore, 1, MPI_FLOAT, k, 0, MPI_COMM_WORLD, &status);
				MPI_Recv (&n, 1, MPI_INT, k, 0, MPI_COMM_WORLD, &status);
				MPI_Recv (&mutationIndex, 1, MPI_INT, k, 0, MPI_COMM_WORLD, &status);
				if(currentScore > bestScore)
				{
					bestScore = currentScore;
					best_offset = n;
					bestMutation = mutationIndex;
				}				
			}
				printf("\n\t\t\tSEQUENCE NUMBER %d", i);
				printf("\n\t\t\tBEST SCORE = %f", bestScore);
				printf("\n\t\t\tBEST MUTATAION = %d", bestMutation);
				printf("\n\t\t\tBEST OFFSET = %d\n", best_offset);
				t2 = MPI_Wtime();
				if (rank == 0 && i == sequenceCount - 1)
					printf("Time Elapsed for the Parallel Algorithm: %lf\n", t2 - t1);
				writeResults(f, i, best_offset, bestMutation, bestScore);
		}	
		else	
		{									
			// send best info to proccess 0 to compare best score of proccess 0 with best score of procces k
			MPI_Send (&bestScore, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			MPI_Send (&best_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send (&bestMutation, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		free(originalSigns);
		freeCudaMemory(seq1, seq2, signs);
		
	}
	free(sequences);
	fclose(f);
    	MPI_Finalize();

	return 0;
}



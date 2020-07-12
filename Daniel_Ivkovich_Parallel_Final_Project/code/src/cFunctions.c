#include "Header.h"




float computeScore(float w[], char* signs)
{
//	Given similarity string: :::**.:**..:.* compute its score

	int matches = 0, conservatives = 0, semiConservatives = 0, miss = 0, i, k;
	k = (int)strlen(signs);
	char c;
	#pragma omp for
	for(i=0; i < k; i++)
	{
		c = signs[i];
		if(c == '*')
			matches ++;
		else if(c == ':')
			conservatives ++;
		else if(c == '.')
			semiConservatives ++;
		else if(c == ' ')
			miss ++;
	}
	float result = w[0] * matches - w[1] * conservatives - w[2] * semiConservatives - w[3] * miss;
	return result;
}

void findBestCombination(int rank, int size, int offset, int mutationIndex, float currentScore, float* bestScore, int* best_offset, int* bestMutation,
							Sequence* currentSequence, char* originalSigns, char* seq1, char* seq2, char* signs, float w[])
{
//	Given a current Mutation Index, Iterate over the relational share of the current proccesses offsets and compute the best offset for that subgroup
	int n;
	#pragma omp for
	for (n = (rank/size) * offset; n < ((rank + 1)/size) * offset; n++)
	{

		GPU_Create_Signs(currentSequence, n, originalSigns, mutationIndex, seq1, seq2, signs);
		currentScore = computeScore(w, originalSigns);

		if (currentScore > *bestScore)
		{
			*bestScore = currentScore;
			*best_offset = n;
			*bestMutation = mutationIndex;
		}

	}
}
void writeResults(FILE* f, int sequenceID, int bestOffset, int bestMutation, float bestScore)
{
//	Given the file f is already open, write one line of the optimal combination regarding one minor sequence of the input
	if (!f)
	{
		printf("FILE IS NOT OPEN\n");
		fflush(stdout);
	}
	fprintf(f, "ID = %d\t Best Offset = %d\t Best Mutation = %d\t Score = %f\n", sequenceID, bestOffset, bestMutation, bestScore);


}

void readAllSequences(MainSequence* ms, Sequence*** sequences, int* sequenceCount)
{
//	Reads the input into data structures: 
//	MainSequence will save the main string and the weights
//	Sequence will save a single minor sequence.

	FILE *f;
	f = fopen(INPUT_FILE_PATH, "r");
	
	if (!f)
	{
		printf("Unable to open file2!\n");
		fflush(stdout);
	}
	
	fscanf(f,"%f %f %f %f", &ms->w[0], &ms->w[1], &ms->w[2], &ms->w[3]);	// Read weights
	fscanf(f, "%s", ms->letters);						// Read the Main String Sequence
	fscanf(f,"%d", sequenceCount);						// Number of minor Sequences
	ms->length = strlen(ms->letters);

	
	*sequences = (Sequence**)malloc((*sequenceCount) * sizeof(Sequence*));	// Alocate memory for the NSQ2 sequences
	int i;
	for(i = 0; i<(*sequenceCount); i++)					// Read each Sequence into the Sequence array
	{
		(*sequences)[i] = (Sequence*)malloc(sizeof(Sequence));
		fscanf(f, "%s", (*sequences)[i]->letters);
		(*sequences)[i]->length = (int)strlen((*sequences)[i]->letters);
	}


	fclose(f);
}

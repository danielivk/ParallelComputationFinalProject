#pragma once

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"



//---------------------------------- Defines ---------------------------------------------------
#define INPUT_FILE_PATH "./input.txt"
#define OUTPUT_FILE_PATH "./output.txt"




//---------------------------------- Structs---------------------------------------------------

typedef struct MainSequence
{
	int length;
	char letters[3000];
	float w[4];
} MainSequence;

typedef struct Sequence
{
	int length;
	char letters[2000];
} Sequence;





//---------------------------------- Method Declerations ---------------------------------------------------




//	Input/Output
void readAllSequences(MainSequence* ms, Sequence*** sequences, int* sequenceCount);
void writeResults(FILE* f, int sequenceID, int bestOffset, int bestMutation, float bestScore);

// 	CPU
float computeScore(float w[], char* signs);
void findBestCombination(int rank, int size, int offset, int mutationIndex, float currentScore, float* bestScore, int* best_offset, int* bestMutation,
Sequence* currentSequence, char* originalSigns, char* seq1, char* seq2, char* signs, float w[]);


//	GPU
int GPU_Create_Signs(Sequence* s, int n, char* originalSigns, int mutationIndex, char* seq1, char* seq2, char* signs);
cudaError_t allocateCudaMemory(char** seq1, char** seq2, char** signs, int msLength, int sLength);
cudaError_t copyInformationToCuda(char* seq1, char* seq2, MainSequence* ms, Sequence* s);
void freeCudaMemory(char* seq1, char* seq2, char* signs);

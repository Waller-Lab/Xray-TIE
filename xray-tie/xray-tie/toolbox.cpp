//	toolbox.cpp
//  orginal author: Diivanand Ramalingam
//  original institution: Computational Optical Imaging Lab at UC Berkeley (Prof. Laura Waller's Lab)

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include "toolbox.h"

using namespace std;

void quitProgramPrompt(bool success)
{
  int c;
  if(success)
	printf( "\nProgram Executed Successfully. Press ENTER to quit program...\n" );
  else
	printf( "\nProgram Execution Failed. Press ENTER to quit program...\n" );
  fflush( stdout );
  do c = getchar(); while ((c != '\n') && (c != EOF));
}

char **getFilenames(char* srcFolder, char* prefix, int start, int end){

	int numFiles = end - start + 1;
	char **filenames = (char **) malloc(sizeof(char *) * numFiles);

	for(int i = 0;i < numFiles; i++) {
		char *numString = (char *) malloc(sizeof(char) * 5);
		itoa(start+i, numString,10);
		filenames[i] = (char *) malloc(sizeof(char) * (6 + strlen(srcFolder) + strlen(prefix) + strlen(numString))); //6 because null character, .tif, and forward slash
		strcpy(filenames[i], srcFolder);
		strcat(filenames[i], "\\");
		strcat(filenames[i], prefix);
		strcat(filenames[i], numString);
		strcat(filenames[i], ".tif");
	}
	return filenames;
}

char *createFullFilename(char* path, char* filename){
	char *output = (char *) malloc(sizeof(char) * (strlen(path) + strlen(filename) + 2));
	strcpy(output, path);
	strcat(output, "\\");
	strcat(output, filename);
	return output;
}

float* toFloatArray(float** image_rows, int width, int height)
{
	float* output = (float *) calloc(width*height, sizeof(float));
	float* buffer = (float *) calloc(width, sizeof(float));
	for(int i = 0; i < height; i++){
		memcpy(buffer, image_rows[i], width*sizeof(float));
		for(int j = 0; j < width; j++){
			output[j+(i*width)] = buffer[j]; 
		}
	}
	return output;
}

float** toFloat2D(float *image, int width, int height){
	float** image_rows = (float**) calloc(height, sizeof(float *));
	image_rows[0] = image;
	for (int i=1; i<height; i++) {
		image_rows[i] = image_rows[i-1] + width;
	}
	return image_rows;
}
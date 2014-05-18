//	toolbox.h
//  orginal author: Diivanand Ramalingam
//  original institution: Computational Optical Imaging Lab at UC Berkeley (Prof. Laura Waller's Lab)


#ifndef TOOLBOX_H
#define TOOLBOX_H


//function declarations
void quitProgramPrompt(bool);
char **getFilenames(char*, char*, int, int);
char *createFullFilename(char*, char*);
float* toFloatArray(float** image, int, int);
float** toFloat2D(float*, int, int);

#endif
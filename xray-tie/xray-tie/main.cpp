//	main.cpp
//  orginal author: Diivanand Ramalingam
//  original institution: Computational Optical Imaging Lab at UC Berkeley (Prof. Laura Waller's Lab)

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <arrayfire.h>
#include "tiff_io-win.h"
#include "toolbox.h"

using namespace af;
void stateArguments(float IinVal, float Mag, float R2, float mu, float delta, float ps, float reg);
void calculateThickness(float *I2, int height, int width, float IinVal, float Mag, float R2, float mu, float delta, float ps, float reg);

/* Height is the number of rows (x) and Width is the number of columns (y)*/
int main(int argc, char **argv)
{
	if(argc != 13){
		printf("Incorrect number of arguments. Usage: ./tie input_folder output_folder prefix start_num end_num Iin Mag R2 mu delta ps reg\n");
		quitProgramPrompt(0);
		return 1;
	}else {
		char *srcFolder = argv[1];
		char *destFolder = argv[2];
		char *prefix = argv[3];
		int start = atoi(argv[4]);
		int end = atoi(argv[5]);
		int numFiles = end - start + 1;
		char **filenames = getFilenames(srcFolder, prefix, start, end);
		char **outfilenames = getFilenames(destFolder, prefix, start, end);
		TIFFSetWarningHandler(NULL);
		//IinVal, Mag, R2, mu, delta, ps
		float IinVal = atof(argv[6]);
		float Mag = atof(argv[7]);
		Mag = 1.0; //Right now algorithm doesn't work for Mag other than 1.0, so for now Mag argument isn't supported.
		float R2 = atof(argv[8]);
		float mu = atof(argv[9]);
		float delta = atof(argv[10]);
		float ps = atof(argv[11]);
		float reg = atof(argv[12]);

		stateArguments(IinVal, Mag, R2, mu, delta, ps, reg);

		TiffIO* tiff_io = new TiffIO();
		int width;
		int height;
		printf("Processing Input Files: \n");
		for(int i = 0;i < numFiles; i++) {
			float **image;
			//read iamge
			image = tiff_io->read16bitImage(filenames[i], &width, &height);

			if(!image){
				printf("Error reading image\n");
			}else {
				//convert image to 1D for CUDA processing
				float *image1D = toFloatArray(image, width, height);
				
				printf("\nProcessing file %s\n", filenames[i]);
				//Process Image
				
				calculateThickness(image1D, height, width, IinVal, Mag, R2, mu, delta, ps, reg);
				
				//End Processing of Image
				//convert image back to 2D for outputting
				image = toFloat2D(image1D, width, height);
				//output image
				printf("\nFile Processed. Outputting to %s\n", outfilenames[i]);
				tiff_io->write16bitImage(image, outfilenames[i], width, height);
				

				//free memory
				delete image;
				free(image1D);
			}
		}
		delete tiff_io;
		quitProgramPrompt(1);
		return 0;
	}
}

void stateArguments(float IinVal, float Mag, float R2, float mu, float delta, float ps, float reg)
{
	std::cout << "Input Argument Values:" << std::endl;
	std::cout << "IinVal: " << IinVal << std::endl;
	std::cout << "Mag: " << Mag << std::endl;
	std::cout << "R2: " << R2 << " mm" << std::endl;
	std::cout << "mu: " << mu << " mm^-1" << std::endl;
	std::cout << "delta: " << delta << std::endl;
	std::cout << "ps: " << ps << " mm" << std::endl;
	std::cout << "reg: " << reg << std::endl;
}

void calculateThickness(float *I2, int height, int width, float IinVal, float Mag, float R2, float mu, float delta, float ps, float reg)
{
	std::cout << "Calculating Thickness" << std::endl;
}
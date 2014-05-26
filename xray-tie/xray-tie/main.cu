//	main.cpp
//  orginal author: Diivanand Ramalingam
//  original institution: Computational Optical Imaging Lab at UC Berkeley (Prof. Laura Waller's Lab)

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include "tiff_io-win.h"
#include "toolbox.h"
#include "pointwise_matrix_ops.h"
#include "fourier_tools.h"

//This constant may already be in <cmath> so I should switch the using those in the code but for now I'm using these
#define PI  3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
//Not all GPUS can handle large block sizes which is why it's only 16 for now
#define BLOCKSIZEX 16 //should increase this to 32,64,128, etc. to see potentially better performance gains!
#define BLOCKSIZEY 16 //should increase this to 32,64,128, etc. to see potentially better performance gains!

//Global time variable
double total_time_elapsed = 0.0;

//device function delclaration
__global__ void xrayTIEHelperKernel(cufftComplex *denominator_dev, float *freq_vector_dev, int N, float R2, float delta, float Mag, float mu, float reg);

//host function declarations
void stateArguments(float IinVal, float Mag, float R2, float mu, float delta, float ps, float reg);
cudaError_t calculateThickness(float *output, float *image, int height, int width, float IinVal, float Mag, float R2, float mu, float delta, float ps, float reg);

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
		//Mag = 1.0; //Right now algorithm doesn't work for Mag other than 1.0, so for now Mag argument isn't supported.
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
				
				float *image_dev = 0;
				float *output = 0;
				
				printf("\nProcessing file %s\n", filenames[i]);
				//Process Image

				//Allocate space on GPU and then transfer the input image to the GPU
				cudaError_t cudaStatus;

				cudaStatus = cudaMalloc((void**)&image_dev, sizeof(float) * height * width);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMalloc for image_dev failed! Error Code: %d", cudaStatus);
				}else {
					cudaStatus = cudaMemcpy(image_dev, image1D, sizeof(float) * width * height, cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy for image_dev failed! Error Code: %d", cudaStatus);
					}else{
						output = (float *) malloc(sizeof(float) * width * height);
						calculateThickness(output, image_dev, height, width, IinVal, Mag, R2, mu, delta, ps, reg);
					}
				}

				//End Processing of Image
				//convert 1D output to 2D for outputting
				float *image1DOut = 0;
				if(output){
					//printMatrix1D("output", output, height, width);
					image1DOut = output;
				}else {
					fprintf(stderr, "\n Output is NULL so using original image as output");
					image1DOut = image1D;
				}
				float **imageOut = toFloat2D(image1DOut, width, height);
				//output image
				printf("\nFile Processed. Outputting to %s\n", outfilenames[i]);
				tiff_io->write16bitImage(imageOut, outfilenames[i], width, height);
			
				//free memory
				free(image1D);
				free(output);
				free(imageOut);
				cudaFree(image_dev);
				delete image;
			}
		}
		delete tiff_io;
		printf("\nTotal time to process %d images: %f seconds\n", numFiles,total_time_elapsed);
		quitProgramPrompt(true);
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

/*
Calculates thickness according to Paganin Phase paper algorithm: http://www.ncbi.nlm.nih.gov/pubmed/12000561
*/
cudaError_t calculateThickness(float* output, float *image_dev, int height, int width, float IinVal, float Mag, float R2, float mu, float delta, float ps, float reg)
{
	cudaError_t cudaStatus;
	cufftResult cufftStatus;
	cufftHandle plan = 0;
	if(height != width){
		fprintf(stderr, "Only works on square matrices whose dimension is a power of two!\n", cudaStatus);
        goto thickness_end;
	}
	//declare and initialize variables used when calling CUDA kernels
	int size = height * width;
	int block_size_x = BLOCKSIZEX;
    int block_size_y = BLOCKSIZEY;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (height/dimBlock.x, width/dimBlock.y);
	int N = width;
	//Handle N not multiple of block_size_x or block_size_y but this shouldn't be the case since N power of 2
	//And blocksize should always be a power of 2 for both correctness and efficiency
    if (height % block_size_x !=0 ) dimGrid.x+=1;
    if (width % block_size_y !=0 ) dimGrid.y+=1;

	std::cout << "Calculating Thickness..." << std::endl;
	std::clock_t begin = std::clock();
	//Code begins here

	//declare device pointers
	float *int_seq_dev = 0;
	float *freq_vector_dev = 0;
	//float *image_dev = 0;
	float *output_dev = 0;
	cufftComplex *image_complex_dev = 0;
	cufftComplex *fft_output_dev = 0;
	cufftComplex *fft_shifted_output_dev = 0;
	cufftComplex *ifft_shifted_input_dev = 0;
	cufftComplex *ifft_output_dev = 0;
	cufftComplex *denominator_dev = 0;

	//Start memory allocation of device vectors and convert/copy input image to complex device vector

	//Allocate memory for 9 device vectors (potential speedup to be gained by reducing the number of device vectors used)
	cudaStatus = cudaMalloc((void**)&int_seq_dev, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for int_seq_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&freq_vector_dev, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for freq_vector_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&output_dev, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for output_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&image_complex_dev, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for image_complex_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&fft_output_dev, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for fft_output_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&fft_shifted_output_dev, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for fft_shifted_output_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&ifft_shifted_input_dev, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for ifft_shifted_input_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&ifft_output_dev, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for ifft_output_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	cudaStatus = cudaMalloc((void**)&denominator_dev, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for denominator_dev failed! Error Code: %d", cudaStatus);
        goto thickness_end;
    }
	
	//scale by magnification
	//printDeviceMatrixValues("img_dev", image_dev,N,N);
	pointwiseRealScaleRealMatrix<<<dimGrid, dimBlock>>>(image_dev, image_dev, Mag*Mag, N, N);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointwiseRealScaleMatrix!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceMatrixValues("img_dev_scaled", image_dev,N,N);
	//convert input image real device vector to complex device vector
	real2complex<<<dimGrid,dimBlock>>>(image_dev, image_complex_dev, N, N);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching real2complex!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceComplexMatrixValues("img_complex_dev", image_complex_dev,N,N);
	//End memory allocation of device vectors and convert/copy input image to complex device vector

	//Start creation of frequency axis
	//generate integer sequence used in creating frequency axis
	genIntSequence<<<1, N>>>(int_seq_dev, 0, N-1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching genIntSequence!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceMatrixValues("int_seq_dev", int_seq_dev,1,N);
	//create omega axis
	pointwiseRealScaleRealMatrix<<<1,N>>>(freq_vector_dev, int_seq_dev, 2*PI/N, N, 1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointwiseRealScaleRealMatrix!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceMatrixValues("omega_freq_vector_dev", freq_vector_dev,1,N);
	//Shift zero to center - for even case, pull back by pi, note N is even by our assumption of powers of 2
	pointwiseAddRealConstantToRealMatrix<<<1,N>>>(freq_vector_dev, freq_vector_dev, -PI, N, 1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addReadConstantToRealMatrix!\n", cudaStatus);
        goto thickness_end;
    }
	//convert to cyclical frequencies (hertz) and scale by pixel size
	pointwiseRealScaleRealMatrix<<<1,N>>>(freq_vector_dev, freq_vector_dev, 1/(2*PI), N, 1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointwiseRealScaleRealMatrix!\n", cudaStatus);
        goto thickness_end;
    }
	pointwiseRealScaleRealMatrix<<<1,N>>>(freq_vector_dev, freq_vector_dev, 1/ps, N, 1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointwiseRealScaleRealMatrix!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceMatrixValues("freq_vector_dev after Pi shift to center", freq_vector_dev,1,N);
	//End creation of frequency axis

	//Fourier Transform image and scale according to Paginin phase algorithm

	cufftPlan2d(&plan, N, N, CUFFT_C2C);
	cufftStatus = cufftExecC2C(plan, image_complex_dev, fft_output_dev, CUFFT_FORWARD);
	if(cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftExecC2C returned error code %d after attempting 2D fft!\n", cufftStatus);
        goto thickness_end;
	}
	//fft shift the spectrum of this signal
	fftShift2D<<<dimGrid,dimBlock>>>(fft_shifted_output_dev, fft_output_dev, width);
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fftShift2D!\n", cudaStatus);
        goto thickness_end;
    }
	
	//printDeviceComplexMatrixValues("fft_shifted_output_dev", fft_shifted_output_dev, N,N);
	pointwiseRealScaleComplexMatrix<<<dimGrid,dimBlock>>>(fft_shifted_output_dev, fft_shifted_output_dev, mu/IinVal, N, N);
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after pointwiseRealScaleComplexMatrix!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceComplexMatrixValues("fft_shifted_output_dev scaled", fft_shifted_output_dev, N,N);
	//End Fourier Transform and scaling

	//Create the denominator shown in the Paganin phase algorithm
	xrayTIEHelperKernel<<<dimGrid, dimBlock>>>(denominator_dev, freq_vector_dev, N, R2, delta, Mag, mu, reg);
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching xrayTIEHelperKernel!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceComplexMatrixValues("denominator_dev", denominator_dev, N,N);
	//End creation of denominator

	//pointwise divide, ifft, pointwise log, and inverse-mu-scaling as shown in Paganin phase algorithm
	//pointwise divide
	pointwiseDivideComplexMatrices<<<dimGrid, dimBlock>>>(fft_shifted_output_dev, fft_shifted_output_dev, denominator_dev, N, N);
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching xrayTIEHelperKernel!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceComplexMatrixValues("inv_term (as mentioned in matlab algorithm)", fft_shifted_output_dev, N,N);
	//ifftshift
	fftShift2D<<<dimGrid, dimBlock>>>(ifft_shifted_input_dev, fft_shifted_output_dev, N);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fftShift2D!\n", cudaStatus);
        goto thickness_end;
    }
	//ifft
	cufftStatus = cufftExecC2C(plan, ifft_shifted_input_dev, ifft_output_dev, CUFFT_INVERSE);
	if(cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftExecC2C returned error code %d after attempting 2D fft!\n", cufftStatus);
        goto thickness_end;
	}
	//normalized and convert to real device vector
	float scale = 1.f / ( (float) height * (float) width );
	//convert complex to real
	complex2real_scaled<<<dimGrid, dimBlock>>>(ifft_output_dev, output_dev, N, N, scale);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching complex2real_scaled!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceMatrixValues("ifftReal", output_dev, N,N);
	//take pointwise log and scale to obtain projected thickness!
	//pointwise natural log
	pointwiseNaturalLogRealMatrix<<<dimGrid, dimBlock>>>(output_dev, output_dev, N, N);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching complex2real_scaled!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceMatrixValues("logOutput (like in matlab algorithm)", output_dev, N,N);
	//pointwise real scale
	pointwiseRealScaleRealMatrix<<<dimGrid, dimBlock>>>(output_dev, output_dev, -(1/mu), N, N);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pointwiseRealScaleRealMatrix!\n", cudaStatus);
        goto thickness_end;
    }
	//printDeviceMatrixValues("output", output_dev, N,N);
	//Transfer output device vector to our host output vector and we are done!
	cudaStatus = cudaMemcpy(output, output_dev, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for output_dev failed!");
        goto thickness_end;
    }

	//destroy cufft plan and free memory allocated on device (FYI device is another name for GPU, host is CPU)
thickness_end:
	cudaFree(int_seq_dev);
	cudaFree(freq_vector_dev);
	cudaFree(output_dev);
	cudaFree(image_complex_dev);
	cudaFree(fft_output_dev);
	cudaFree(fft_shifted_output_dev);
	cudaFree(ifft_shifted_input_dev);
	cudaFree(ifft_output_dev);
	cudaFree(denominator_dev);
	if(plan)
		cufftDestroy(plan);

	//Code ends here
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "\nDone. Took " << elapsed_secs << " seconds" << std::endl;
	total_time_elapsed += elapsed_secs;
	return cudaStatus;
}

//computes the denominators seen in Paganin phase algorithm paper
__global__
void xrayTIEHelperKernel(cufftComplex *denominator_dev, float *freq_vector_dev, int N, float R2, float delta, float Mag, float mu, float reg)
{
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < N && idy < N) {
        int index = idx + idy*N;
		denominator_dev[index].x = (R2*delta)*(((freq_vector_dev[idx]*freq_vector_dev[idx]) + (freq_vector_dev[idy]*freq_vector_dev[idy]))/Mag) + mu + reg;
		denominator_dev[index].y = 0;
    }
}
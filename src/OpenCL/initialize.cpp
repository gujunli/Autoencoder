/*
 * Copyright (c) 2013 AMD
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERHCCES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * Author:  Maohua Zhu
 *          Junli Gu
 */

#include "autoencoder.h"
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>

void loadKernelSource(string filename, char* source){
	ifstream fin;
	string srt;
	fin.open(filename, ios_base::in);
	char t[500];
	while(fin.getline(t, 500)){
		srt.append(t);
		srt.push_back('\n');
	}
	for(int i = 0; i < srt.length(); i++)
		source[i] = srt[i];
	source[srt.length()] = 0;
	return;
}

void autoencoder::initialize(traindata* TrainData, vector<int> layersizes, int miniBatchSize){

	batchSize = miniBatchSize;
	trainData = TrainData;
	totalWeightSize = 0;

	initializeGPU(cl_env);

	// set layer sizes
	layersize.push_back(trainData->featureNum);
	for(int l = 0; l < layersizes.size(); l++){
		layersize.push_back(layersizes[l]);
	}
	for(int l = layersizes.size() - 2; l >= 0; l--)
		layersize.push_back(layersizes[l]);
	layersize.push_back(trainData->featureNum);

	layerNum = layersize.size();

	// initialize layer structure
	for(int l = 0; l < layersize.size(); l++){
		int length = layersize[l] * batchSize;
		floatType* v = new floatType[length];
		setFloat(v, 0.0, length);
		ac.push_back(v);
		
		// device buffer
		cl_mem d_v = clCreateBuffer(cl_env.ctx, CL_MEM_READ_WRITE, length * sizeof(floatType), NULL, &cl_env.err);
		d_ac.push_back(d_v);
	}

	// derivative of each layer
	for(int l = 0; l < layersize.size(); l++){
		int length = layersize[l] * batchSize;
		floatType* v = new floatType[length];
		setFloat(v, 0.0, length);
		er.push_back(v);

		// device buffer
		cl_mem d_v = clCreateBuffer(cl_env.ctx, CL_MEM_READ_WRITE, length * sizeof(floatType), NULL, &cl_env.err);
		d_er.push_back(d_v);
	}


	// initialize weight structure
	// calculate the total number of weights
	for(int l = 0; l < layersize.size() / 2; l++){
		int length = layersize[l] * layersize[l + 1];
		totalWeightSize += length;
	}

	for(int l = 0; l < layersize.size() - 1; l++){
		int length= layersize[l + 1];
		totalWeightSize += length;
	}

	W = new floatType[totalWeightSize];
	// device buffer
	d_W = clCreateBuffer(cl_env.ctx, CL_MEM_READ_WRITE, totalWeightSize * sizeof(floatType), NULL, &cl_env.err);

	int offset = 0;

	for(int l = 0; l < layersize.size() / 2; l++){
		int length = layersize[l] * layersize[l + 1];
		floatType* w = W + offset;
		setFloat(w, 0.0, length);
		weight.push_back(w);

		// device sub buffer
		cl_buffer_region region;
		region.origin = offset * sizeof(floatType);
		region.size = length * sizeof(floatType);
		cl_mem d_w = clCreateSubBuffer(d_W, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_env.err);
		d_weight.push_back(d_w);

		offset += length;

		length = layersize[l + 1];
		floatType* bias = W + offset;
		setFloat(bias, 0.0, length);
		b.push_back(bias);

		// device sub buffer
		region.origin = offset * sizeof(floatType);
		region.size = length * sizeof(floatType);
		cl_mem d_bias = clCreateSubBuffer(d_W, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_env.err);
		d_b.push_back(d_bias);

		offset += length;
	}

	for(int l = layersize.size() / 2; l < layersize.size() - 1; l++){
		int length = layersize[l + 1];
		floatType* bias = W + offset;
		setFloat(bias, 0.0, length);
		b.push_back(bias);

		// device sub buffer
		cl_buffer_region region;
		region.origin = offset * sizeof(floatType);
		region.size = length * sizeof(floatType);
		cl_mem d_bias = clCreateSubBuffer(d_W, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_env.err);
		d_b.push_back(d_bias);

		offset += length;
	}

	// set random seed
	// srand(time(NULL));
	srand(37);

	for(int l = 0; l < weight.size(); l++){
		floatType r = sqrt(6) / sqrt(layersize[l] + layersize[l + 1]);
		for(int i = 0; i < layersize[l] * layersize[l + 1]; i++)
			weight[l][i] = ((double)rand() / (double)RAND_MAX) * 2 * r - r;
	}

	// grad of W and B
	Grad = new floatType[totalWeightSize];
	d_Grad = clCreateBuffer(cl_env.ctx, CL_MEM_READ_WRITE, totalWeightSize * sizeof(floatType), NULL, &cl_env.err);

	offset = 0;

	for(int l = 0; l < layersize.size() / 2; l++){
		int length = layersize[l] * layersize[l + 1];
		floatType* w = Grad + offset;
		setFloat(w, 0.0, length);
		Wgrad.push_back(w);

		// device sub buffer
		cl_buffer_region region;
		region.origin = offset * sizeof(floatType);
		region.size = length * sizeof(floatType);
		cl_mem d_w = clCreateSubBuffer(d_Grad, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_env.err);
		d_Wgrad.push_back(d_w);

		offset += length;

		length = layersize[l + 1];
		floatType* bias = Grad + offset;
		setFloat(bias, 0.0, length);
		Bgrad.push_back(bias);

		// device sub buffer
		region.origin = offset * sizeof(floatType);
		region.size = length * sizeof(floatType);
		cl_mem d_bias = clCreateSubBuffer(d_Grad, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_env.err);
		d_Bgrad.push_back(d_bias);

		offset += length;
	}

	for(int l = layersize.size() / 2; l < layersize.size() - 1; l++){
		int length = layersize[l + 1];
		floatType* bias = Grad + offset;
		setFloat(bias, 0.0, length);
		Bgrad.push_back(bias);

		// device sub buffer
		cl_buffer_region region;
		region.origin = offset * sizeof(floatType);
		region.size = length * sizeof(floatType);
		cl_mem d_bias = clCreateSubBuffer(d_Grad, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_env.err);
		d_Bgrad.push_back(d_bias);

		offset += length;
	}

	cl_env.err = clEnqueueWriteBuffer(cl_env.queue, d_W, CL_TRUE, 0, totalWeightSize * sizeof(floatType), (void*)W, 0, NULL, NULL);

	// build OpenCL kernels
	char* source = new char[KERNEL_SOURCE_LENGTH];
	loadKernelSource("cl_autoencoder_kernel.cl", source);
	cl_env.prog = clCreateProgramWithSource(cl_env.ctx, 1, (const char**)&source, NULL, &cl_env.err);

	cl_env.err = clBuildProgram(cl_env.prog, 0, NULL, NULL, NULL, NULL);

	if (cl_env.err == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(cl_env.prog, cl_env.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(cl_env.prog, cl_env.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}

	return;
}

void initializeGPU(CL_Env& cl_env){
	// initialize the OpenCL environment
	cl_env.platform = 0;
	cl_env.device = 0;
	cl_env.props[0] = CL_CONTEXT_PLATFORM;
	cl_env.props[1] = cl_env.props[2] = 0;
	cl_env.ctx = 0;
	cl_env.queue = 0;
	cl_env.event = NULL;
	cl_env.order = clAmdBlasColumnMajor;

	// setup OpenCL environment
	cl_env.err = clGetPlatformIDs(1, &cl_env.platform, NULL);
	if (cl_env.err != CL_SUCCESS) {
		printf( "clGetPlatformIDs() failed with %d\n", cl_env.err );
		return;
	}

	cl_device_id device[10];
	cl_uint devices_n;

	cl_env.err = clGetDeviceIDs(cl_env.platform, CL_DEVICE_TYPE_GPU, 10, device, &devices_n);
	if (cl_env.err != CL_SUCCESS) {
		printf( "clGetDeviceIDs() failed with %d\n", cl_env.err );
		return;
	}

	for(int i = 0; i < devices_n; i++){
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
		printf("  --Device: %d --\n", i);
		clGetDeviceInfo(device[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_NAME = %s\n", buffer);
		clGetDeviceInfo(device[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VENDOR = %s\n", buffer);
		clGetDeviceInfo(device[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VERSION = %s\n", buffer);
		clGetDeviceInfo(device[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(device[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(device[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	}

	cl_env.device = device[1];

	cl_env.props[1] = (cl_context_properties)cl_env.platform;
	cl_env.ctx = clCreateContext(cl_env.props, 1, &cl_env.device, NULL, NULL, &cl_env.err);
	if (cl_env.err != CL_SUCCESS) {
		printf( "clCreateContext() failed with %d\n", cl_env.err );
		return;
	}

	cl_env.queue = clCreateCommandQueue(cl_env.ctx, cl_env.device, CL_QUEUE_PROFILING_ENABLE, &cl_env.err);
	if (cl_env.err != CL_SUCCESS) {
		printf( "clCreateCommandQueue() failed with %d\n", cl_env.err );
		clReleaseContext(cl_env.ctx);
		return;
	}

	/* Setup clAmdBlas. */
	cl_env.err = clAmdBlasSetup();
	if (cl_env.err != CL_SUCCESS) {
		printf("clAmdBlasSetup() failed with %d\n", cl_env.err);
		clReleaseCommandQueue(cl_env.queue);
		clReleaseContext(cl_env.ctx);
		return;
	}

}
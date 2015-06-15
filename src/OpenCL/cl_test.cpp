#include "autoencoder.h"
#include <string>
#include <sstream>

#define KERNEL_SOURCE_LENGTH 20000

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

void cl_test(){
	cl_int ciErrNum;

	// Use the first platform
	cl_platform_id platform;
	ciErrNum = clGetPlatformIDs(1, &platform, NULL);

	// Use the first device
	cl_device_id device;
	ciErrNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	
	cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

	// Create the context
	cl_context ctx = clCreateContext(cps, 1, &device, NULL, NULL, &ciErrNum);

	// Create the command queue
	cl_command_queue myqueue = clCreateCommandQueue(ctx, device, 0, &ciErrNum);

	char* str = new char[20];
	str = "GdkknVnqkc";
	char* outputStr = new char[20];

	// allocate buffer for device
	cl_mem bufferStr = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 20 * sizeof(char), NULL, &ciErrNum);
	cl_mem bufferOutput = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 20 * sizeof(char), NULL, &ciErrNum);

	// copy from host to device
	ciErrNum = clEnqueueWriteBuffer(myqueue, bufferStr, CL_TRUE, 0, 10 * sizeof(char), (void*)str, 0, NULL, NULL);

	// runtime kernel compiling
	char* source = new char[KERNEL_SOURCE_LENGTH];
	loadKernelSource("cl_test_kernel.cl", source);
	cl_program myprog = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, &ciErrNum);

	ciErrNum = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);

	if (ciErrNum == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(myprog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(myprog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}


	cl_kernel mykernel = clCreateKernel(myprog, "hello", &ciErrNum);

	int n = 10;

	clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void*)&bufferStr);
	clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void*)&bufferOutput);
	clSetKernelArg(mykernel, 2, sizeof(int), (void*)&n);

	size_t globalws[1] = {n};

	// Execute the kernel
	ciErrNum = clEnqueueNDRangeKernel(myqueue, mykernel, 1, NULL, globalws, NULL, 0, NULL, NULL);

	// Read the output data back to the host
	ciErrNum = clEnqueueReadBuffer(myqueue, bufferOutput, CL_TRUE, 0, n * sizeof(char), (void*)outputStr, 0, NULL, NULL);

	for(int i = 0; i < n; i++)
		cout << outputStr[i];
	cout << endl;

	return;
}


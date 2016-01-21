/*
 * Author:  Maohua Zhu
 *          Junli Gu
 */

#ifndef _AUTOENCODER_H_
#define _AUTOENCODER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <acml.h>
#include <clAmdblas.h>
#include <Windows.h>
#include <mmsystem.h>
#include "lbfgs.h"

#pragma comment(lib, "Winmm.lib")
#define BLOCK_SIZE 16
#define KERNEL_SOURCE_LENGTH 20000

using namespace std;

typedef float floatType;

typedef enum{ bp, linear, sigmoid, tanhAct } activation;
typedef enum{ CPU, GPU } Platform;

typedef struct{
	unsigned featureNum;
	unsigned size;
	byte* label;
	floatType* data;
}traindata;

class CL_Env{
public:
	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context_properties props[3];
	cl_context ctx;
	cl_command_queue queue;
	cl_event event;
	cl_program prog;
	clAmdBlasOrder order;
};

class autoencoder{
protected:
	CL_Env cl_env;
	cl_kernel addBias, tanhActivation;

	vector<cl_mem> d_ac, d_b, d_weight, d_Bgrad, d_Wgrad, d_er;
	cl_mem d_W, d_Grad;
	vector<floatType*> ac, b, weight, Bgrad, Wgrad, er;
	floatType* W, *Grad;
	vector<int> layersize;
	traindata* trainData;
	Platform plat;
	int layerNum;
	int batchSize;
	int totalWeightSize;
	int startIndex;
	int maxIter;
public:
	autoencoder(){};
	void initialize(traindata* trainData, vector<int> layersize, int miniBatchSize);
	void loadWeightFromFile(string filename);
	floatType train(int startIndex, int maxIter, traindata* data);
	floatType test(traindata* data){return 0.0;};
	floatType compute(void);
	floatType computeGPU(void);
	floatType computePlatform(void);
	void reconstruct(void);
	void fetchDeviceData(void);
	void storeDeviceData(void);
	void writeToFile(void){};
	void setPlatform(Platform p){ this->plat = p; };
	void forwardPropNoTransLinear(cl_mem weight,
		cl_mem input,
		cl_mem output,
		int inputLayerSize,
		int outputLayerSize,
		int batchSize);
	void forwardPropNoTransTanh(cl_mem weight,
		cl_mem input,
		cl_mem output,
		int inputLayerSize,
		int outputLayerSize,
		int batchSize);

protected:
	static lbfgsfloatval_t _evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
    {
		return reinterpret_cast<autoencoder*>(instance)->evaluate(x, g, n, step);
    }

    lbfgsfloatval_t evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
    {
		lbfgsfloatval_t fx;
		fx = computePlatform();
		memcpy(g, Grad, totalWeightSize * sizeof(floatType));
        return fx;
    }

    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
		return reinterpret_cast<autoencoder*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
        printf("Iteration %d:\n", k);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
        printf("\n");
        return 0;
    }

};


traindata* loadData(string datapath);
traindata* loadData(string datapath, int expandFactor);
traindata* loadData_txt(string datapath);
void loadKernelSource(string filename, char* source);
void tanhActivate(CL_Env cl_env, cl_mem ac, int arrayLength, cl_event* event);
void addBiasValue(CL_Env cl_env, cl_mem bias, cl_mem ac, int arrayLength, int batchSize, cl_event* event);

void cl_test();

void optimizeAutoencoderLBFGS(vector<int> layersizes, string datapath, floatType finalObjective, Platform plat, int batchSize);
void testAutoencoder(vector<int> layersizes, string datapath, floatType finalObjective, Platform plat, int batchSize);

void initializeGPU(CL_Env& cl_env);

void printLog(string filename, floatType* data, int size, int dimA, int dimB);

void setFloat(floatType* data, int val, int size);

#endif

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

void matrixMul(floatType* dst, floatType* derDst, floatType* A, floatType* B, int rowA, int colB, int stride, bool Ta, activation act);
void matrixMulBack(floatType* dst, floatType* outderv, floatType* h, int dervLength, int hLength, int batchSize, bool Tdst);

floatType autoencoder::train(int start, int maxiter, traindata* traindata){
	startIndex = start;
	maxIter = maxiter;

	lbfgs_parameter_t* param = new lbfgs_parameter_t;
	floatType fx = 0;
	
	param->m = 6;
	param->epsilon = 1e-5;
	param->past = 0;
	param->delta = 1e-5;
	param->max_iterations = 20;
	param->linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
	param->max_linesearch = 40;
	param->min_step = 1e-20;
	param->max_step = 1e+20;
	param->ftol = 1e-4;
	param->gtol = 1e-3;
	param->xtol = 1e-9;
	param->wolfe = 1e-3;
	param->orthantwise_c = 0.0;
	param->orthantwise_start = 0;
	param->orthantwise_end = -1;
	
	// load input data to the first layer
	for(int i = 0; i < batchSize; i++){
		for(int j = 0; j < layersize[0]; j++){
			ac[0][i * layersize[0] + j] = trainData->data[(i + startIndex) * layersize[0] + j];
		}
	}



	int ret = lbfgs(totalWeightSize, W, &fx, _evaluate, _progress, this, param);
	printf("L-BFGS optimization terminated with status code = %d\n", ret);

	return fx;
}



// compute cost function and gradients
floatType autoencoder::compute(){

	// forward prop
	for(int l = 0; l < layerNum - 1; l++){
		for(int i = 0; i < batchSize; i++)
			for(int j = 0; j < layersize[l + 1]; j++)
				ac[l + 1][i * layersize[l + 1] + j] = b[l][j];
		if(l < weight.size()){
			dgemm('n', 'n', layersize[l + 1], batchSize, layersize[l], 1.0, weight[l], layersize[l + 1], ac[l], layersize[l], 1.0, ac[l + 1], layersize[l + 1]);
			// matrixMul(ac[l], er[l], weight[l - 1], ac[l - 1], layersize[l], batchSize, layersize[l - 1], false, tanhAct);
		}
		else{
			dgemm('t', 'n', layersize[l + 1], batchSize, layersize[l], 1.0, weight[2 * weight.size() - l - 1], layersize[l], ac[l], layersize[l], 1.0, ac[l + 1], layersize[l + 1]);
			// matrixMul(ac[l], er[l], weight[layersize.size() - 1 - l], ac[l - 1], layersize[l], batchSize, layersize[l - 1], true, tanhAct);
		}
		if(l != layerNum - 2){
			for(int i = 0; i < batchSize; i++)
				for(int j = 0; j < layersize[l + 1]; j++)
					ac[l + 1][i * layersize[l + 1] + j] = tanh(ac[l + 1][i * layersize[l + 1] + j]);
		}

	}

	// compute cost
	for(int i = 0; i < batchSize; i++)
		for(int j = 0; j < layersize[layerNum - 1]; j++)
			er[layerNum - 1][i * layersize[layerNum - 1] + j] = ac[layerNum - 1][i * layersize[layerNum - 1] + j] - ac[0][i * layersize[layerNum - 1] + j];
	floatType cost = 0;
	for(int i = 0; i < batchSize * layersize[layerNum - 1]; i++)
		cost += er[layerNum - 1][i] * er[layerNum - 1][i];
	cost /= 2 * trainData->featureNum;

	// backprop
	for(int i = 0; i < batchSize * layersize[layerNum - 1]; i++)
		er[layerNum - 1][i] = er[layerNum - 1][i] / layersize[layerNum - 1];

	for(int l = layerNum - 1; l > 0; l--){
		if(l > 1){
			// compute error
			if(l > weight.size())
				dgemm('n', 'n', layersize[l - 1], batchSize, layersize[l], 1.0, weight[2 * weight.size() - l], layersize[l - 1], er[l], layersize[l], 0.0, er[l - 1], layersize[l - 1]);
				// matrixMulBack(Wgrad[2 * weight.size() - ll], er[ll], ac[ll - 1], layersize[ll], layersize[ll - 1], batchSize, true);
			else
				dgemm('t', 'n', layersize[l - 1], batchSize, layersize[l], 1.0, weight[l - 1], layersize[l], er[l], layersize[l], 0.0, er[l - 1], layersize[l - 1]);
				// matrixMulBack(Wgrad[ll - 1], er[ll], ac[ll - 1], layersize[ll], layersize[ll - 1], batchSize, false);

			for(int i = 0; i < batchSize; i++){
				for(int j = 0; j < layersize[l - 1]; j++){
					er[l - 1][i * layersize[l - 1] + j] *= (1 - ac[l - 1][i * layersize[l - 1] + j]) * (1 + ac[l - 1][i * layersize[l - 1] + j]);
				}

			}
		}
		// bias gradient
		for(int i = 0; i < layersize[l]; i++){
			floatType t = 0;
			for(int k = 0; k < batchSize; k++)
				t += er[l][k * layersize[l] + i];
			Bgrad[l - 1][i] = t;
		}

		// compute weight gradient
		if(l > weight.size())
			dgemm('n', 't', layersize[l - 1], layersize[l], batchSize, 1.0, ac[l - 1], layersize[l - 1], er[l], layersize[l], 0.0, Wgrad[2 * weight.size() - l], layersize[l - 1]);
			// matrixMul(er[ll - 1], NULL, weight[2 * weight.size() - ll], er[ll], layersize[ll - 1], batchSize, layersize[ll], false, bp);
		else
			dgemm('n', 't', layersize[l], layersize[l - 1], batchSize, 1.0, er[l], layersize[l], ac[l - 1], layersize[l - 1], 1.0, Wgrad[l - 1], layersize[l]);
	}

	return cost;
}

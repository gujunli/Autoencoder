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

void autoencoder::initialize(traindata* TrainData, vector<int> layersizes, int miniBatchSize){

	batchSize = miniBatchSize;
	trainData = TrainData;
	totalWeightSize = 0;


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
		memset(v, 0.0, length * sizeof(floatType));
		ac.push_back(v);
	}

	// derivative of each layer
	for(int l = 0; l < layersize.size(); l++){
		int length = layersize[l] * batchSize;
		floatType* v = new floatType[length];
		memset(v, 0.0, length * sizeof(floatType));
		er.push_back(v);
	}


	// initialize weight structure
	// calculate the total number of weights
	for(int l = 0; l < layersize.size() / 2; l++){
		int length = layersize[l] * layersize[l + 1];
		totalWeightSize += length;
	}

	for(int l = 0; l < layersize.size() - 1; l++){
		int length = layersize[l + 1];
		totalWeightSize += length;
	}

	W = new floatType[totalWeightSize];
	int offset = 0;

	for(int l = 0; l < layersize.size() / 2; l++){
		int length = layersize[l] * layersize[l + 1];
		floatType* w = W + offset;
		memset(w, 0.0, length * sizeof(floatType));
		weight.push_back(w);
		offset += length;
		length = layersize[l + 1];
		floatType* bias = W + offset;
		memset(bias, 0.0, length * sizeof(floatType));
		b.push_back(bias);
		offset += length;
	}

	for(int l = layersize.size() / 2; l < layersize.size() - 1; l++){
		int length = layersize[l + 1];
		floatType* bias = W + offset;
		memset(bias, 0.0, length * sizeof(floatType));
		b.push_back(bias);
		offset += length;
	}

	// set random seed
	srand(time(NULL));

	for(int l = 0; l < weight.size(); l++){
		floatType r = sqrt(6) / sqrt(layersize[l] + layersize[l + 1]);
		for(int i = 0; i < layersize[l] * layersize[l + 1]; i++)
			weight[l][i] = ((double)rand() / (double)RAND_MAX) * 2 * r - r;
	}

	// grad of W and B
	Grad = new floatType[totalWeightSize];
	offset = 0;

	for(int l = 0; l < layersize.size() / 2; l++){
		int length = layersize[l] * layersize[l + 1];
		floatType* w = Grad + offset;
		memset(w, 0.0, length * sizeof(floatType));
		Wgrad.push_back(w);
		offset += length;
		length = layersize[l + 1];
		floatType* bias = Grad + offset;
		memset(bias, 0.0, length * sizeof(floatType));
		Bgrad.push_back(bias);
		offset += length;
	}

	for(int l = layersize.size() / 2; l < layersize.size() - 1; l++){
		int length = layersize[l + 1];
		floatType* bias = Grad + offset;
		memset(bias, 0.0, length * sizeof(floatType));
		Bgrad.push_back(bias);
		offset += length;
	}

	return;
}

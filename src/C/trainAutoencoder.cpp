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

int main(void){
	// this should be much lower for evaluation
	floatType finalObjective = 0.1;

	string datasetpath = "traindata_small.data";

	// two layers [2 * 3072 100] as default
	// modify the parameter to fit your data
	vector<int> layersizes;
	layersizes.push_back(2 * 200);
	layersizes.push_back(100);
	
	// run the training
	optimizeAutoencoderLBFGS(layersizes, datasetpath, finalObjective);
	system("pause");                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	return 0;
}

void optimizeAutoencoderLBFGS(vector<int> layersizes, string datapath, floatType finalObjective){

	cout << "Now Loading data..." << endl;
	traindata* trainData = loadData(datapath);
	cout << "Done." << endl;
	
	// parameters for main loop
	int miniBatchSize = 1000;
	int maxIter = 2000;

	autoencoder nn;
	nn.initialize(trainData, layersizes, miniBatchSize);

	// random permutation attached to the end of training data not yet implemented



	for(int i = 0; i < maxIter; i++){
		int startIndex = (i * miniBatchSize) % trainData->size;
		cout << "startIndex = " << startIndex << ", endIndex = " << startIndex + miniBatchSize - 1 << endl;
		floatType obj = nn.train(startIndex, maxIter, trainData);
		cout << "cost = " << obj << endl;
		if(obj <= finalObjective){
			floatType trainError = nn.test(trainData);
			if(trainError <= finalObjective)
				break;
		}
	}

	nn.writeToFile();

	system("pause");
	return;
}
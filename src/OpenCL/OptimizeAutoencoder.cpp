#include "autoencoder.h"

int main(void){
	// this should be much lower for evaluation
	floatType finalObjective = 0.7;

	string datasetpath = "traindata_small.dat";

	// two layers [2 * 3072 100] as default
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
	int miniBatchSize = 10;
	int maxIter = 20;

	autoencoder nn;
	nn.initialize(trainData, layersizes, miniBatchSize);

	// random permutation attached to the end of training data not yet implemented



	for(int i = 0; i < maxIter; i++){
		int startIndex = (i * miniBatchSize) % trainData->size;
		cout << "startIndex = " << startIndex << ", endIndex = " << startIndex + miniBatchSize - 1 << endl;
		floatType obj = nn.train(startIndex, maxIter, trainData);
		if(obj <= finalObjective){
			floatType trainError = nn.test(trainData);
			if(trainError <= finalObjective)
				break;
		}
	}

	nn.writeToFile();

	return;
}
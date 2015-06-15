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


traindata* loadData(string datapath){
	ifstream fin;
	fin.open(datapath, ios_base::binary);

	traindata* d = new traindata;
	if(d == NULL){
		cerr << "LoadData: Memory Allocation FAILED!" << endl;
	}
	// default data size
	// later to be modified to read from file

	d->featureNum = 3072;
	d->size = 10000;

	d->data = new floatType[d->featureNum * d->size];
	d->label = new byte[d->size];

	// read data from file
	for(int index = 0; index < d->size; index++){
		byte label;
		fin.read((char*)&label, 1);
		d->label[index] = label;
		byte temp[4096];
		fin.read((char*)temp, d->featureNum);
		for(int feature = 0; feature < d->featureNum; feature++){
			d->data[index * d->featureNum + feature] = (floatType)temp[feature] / 256.0;
		}
	}
	fin.close();
	return d;
}

traindata* loadData_txt(string datapath){
	ifstream fin;
	fin.open(datapath, ios_base::binary);

	traindata* d = new traindata;
	if(d == NULL){
		cerr << "LoadData: Memory Allocation FAILED!" << endl;
	}
	// default data size
	// later to be modified to read from file

	d->featureNum = 3072;
	d->size = 10000;

	d->data = new floatType[d->featureNum * d->size];

	// read data from file

	for(int feature = 0; feature < d->featureNum; feature++){
		for(int index = 0; index < d->size; index++){
			double temp;
			fin >> temp;
			d->data[index * d->featureNum + feature] = temp;
		}

	}

	fin.close();
	return d;
}

traindata* loadData(string datapath, int expandFactor){
	ifstream fin;
	fin.open(datapath, ios_base::binary);

	traindata* d = new traindata;
	if(d == NULL){
		cerr << "LoadData: Memory Allocation FAILED!" << endl;
	}
	// default data size
	// later to be modified to read from file

	d->featureNum = 3072 * expandFactor * expandFactor;
	d->size = 10000;

	d->data = new floatType[d->featureNum * d->size];

	// read data from file
	for(int index = 0; index < d->size; index++){
		byte label;
		fin >> label;
		for(int feature = 0; feature < d->featureNum / expandFactor / expandFactor; feature++){
			byte temp;
			fin >> temp;
			d->data[index * d->featureNum + feature * expandFactor + ((feature % 1024) / 32) * 32 * expandFactor] = (floatType)temp / 256.0;
			if(feature % 1024 >= 32){
				for(int i = 1; i < expandFactor; i++){
					d->data[index * d->featureNum + feature * expandFactor + ((feature % 1024 - 32 * i) / 32) * 32 * expandFactor] = 
						d->data[index * d->featureNum + feature * expandFactor + ((feature % 1024) / 32) * 32 * expandFactor] * (expandFactor - i) / expandFactor +
						d->data[index * d->featureNum + (feature - 32) * expandFactor + ((feature % 1024 - 32) / 32) * 32 * expandFactor] * i / expandFactor;
				}
			}
			if(feature % 32 > 0){
				for(int i = 1; i < expandFactor; i++){
					d->data[index * d->featureNum + feature * expandFactor + ((feature % 1024) / 32) * 32 * expandFactor - i] = 
						d->data[index * d->featureNum + feature * expandFactor + ((feature % 1024) / 32) * 32 * expandFactor] * (expandFactor - i) / expandFactor +
						d->data[index * d->featureNum + (feature - 1) * expandFactor + ((feature % 1024) / 32) * 32 * expandFactor] * i / expandFactor;
				}
			}
			if(feature % 1024 >= 32 && feature %32 > 0){
				for(int i = 1; i < expandFactor; i++){
					d->data[index * d->featureNum + feature * expandFactor + ((feature % 1024 - 32 * i) / 32) * 32 * expandFactor - i] = 
						d->data[index * d->featureNum + feature * expandFactor + ((feature % 1024) / 32) * 32 * expandFactor] * (expandFactor - i) / expandFactor +
						d->data[index * d->featureNum + (feature - 32 - 1) * expandFactor + ((feature % 1024 - 32) / 32) * 32 * expandFactor] * i / expandFactor;
				}
			}
		}
	}

	fin.close();
	return d;
}
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


// OpenCL Autoencoder Kernel
#define BLOCKSIZE 16

typedef float floatType;

__kernel void addBias_kernel(
	__global floatType* ac,
	__global floatType* bias,
	int batchSize,
	int layerSize
	){
	int index = get_global_id(0);
	int size = get_global_size(0);
	for(int i = index; i < layerSize; i += size){
		floatType t = bias[i];
		for(int j = 0; j < batchSize; j++){
			ac[i + j * layerSize] = t;
		}
	}
}

__kernel void addBiasValue_kernel(
	__global floatType* src,
	__global floatType* dst,
	int arraySize,
	int batchSize
	){
	int index = get_global_id(0);
	int size = get_global_size(0);
	for(int i = index; i < arraySize; i += size){
		for(int j = 0; j < batchSize; j++){
			dst[j * arraySize + i] = src[i];
		}
	}
}


__kernel void tanhActivation_kernel(
	__global floatType* input,
	int arrayLength
	){
	int index = get_global_id(0);
	int size = get_global_size(0);
	for(int i = index; i < arrayLength; i += size){
		input[i] = tanh(input[i]);
	}

}

__kernel void sub_kernel(
	__global floatType* dst,
	__global floatType* a,
	__global floatType* b,
	int layerSize,
	int arrayLength
	){
	int index = get_global_id(0);
	int size = get_global_size(0);
	for(int i = index; i < arrayLength; i += size){
		dst[i] = (a[i] - b[i]) / (floatType)layerSize;
	}
}

__kernel void derive_kernel(
	__global floatType* input,
	__global floatType* ac,
	int arrayLength
	){
	int index = get_global_id(0);
	int size = get_global_size(0);
	for(int i = index; i < arrayLength; i += size){
		input[i] *= (1 + ac[i]) * (1 - ac[i]);
	}
}

__kernel void computeBiasGrad_kernel(
	__global floatType* biasGrad,
	__global floatType* er,
	int layerSize,
	int batchSize
	){
	int index = get_global_id(0);
	int size = get_global_size(0);
	for(int i = index; i < layerSize; i += size){
		floatType t = 0;
		for(int k = 0; k < batchSize; k++)
			t += er[k * layerSize + i];
		biasGrad[i] = t;
	}
}


__kernel void forwardPropNoTransLinear_kernel(
	__global floatType* weight,
	__global floatType* input,
	__global floatType* output,
	int inputLayerSize,
	int outputLayerSize,
	int batchSize
	){
    
	int blockRow = get_group_id(0);
    int blockCol = get_group_id(1);
    
    int outputOffset = outputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * blockRow;
    
    // the output value
    floatType outputValue = 0.0;
    
    int row = get_local_id(0);
    int col = get_local_id(1);
    
    // main loop
    for(int m = 0; m < (inputLayerSize / BLOCKSIZE); m++){
        
        int weightOffset = outputLayerSize * BLOCKSIZE * m + BLOCKSIZE* blockRow;
        
        int inputOffset = inputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * m;
        
        __local floatType s_weight[BLOCKSIZE][BLOCKSIZE];
        __local floatType s_input[BLOCKSIZE][BLOCKSIZE];
        
        s_weight[row][col] = weight[weightOffset + col * outputLayerSize + row];
        s_input[row][col] = input[inputOffset + col * inputLayerSize + row];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int i = 0; i < BLOCKSIZE; i++)
            outputValue += s_weight[row][i] * s_input[i][col];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    output[outputOffset + col * outputLayerSize + row] += outputValue;
    
}

__kernel void forwardPropNoTransTanh_kernel(
	__global floatType* weight,
	__global floatType* input,
	__global floatType* output,
	int inputLayerSize,
	int outputLayerSize,
	int batchSize
	){
    
	int blockRow = get_group_id(0);
    int blockCol = get_group_id(1);
    
    int outputOffset = outputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * blockRow;
    
    // the output value
    floatType outputValue = 0.0;
    
    int row = get_local_id(0);
    int col = get_local_id(1);
    
    // main loop
    for(int m = 0; m < (inputLayerSize / BLOCKSIZE); m++){
        
        int weightOffset = outputLayerSize * BLOCKSIZE * m + BLOCKSIZE* blockRow;
        
        int inputOffset = inputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * m;
        
        __local floatType s_weight[BLOCKSIZE][BLOCKSIZE];
        __local floatType s_input[BLOCKSIZE][BLOCKSIZE];
        
        s_weight[row][col] = weight[weightOffset + col * outputLayerSize + row];
        s_input[row][col] = input[inputOffset + col * inputLayerSize + row];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int i = 0; i < BLOCKSIZE; i++)
            outputValue += s_weight[row][i] * s_input[i][col];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    output[outputOffset + col * outputLayerSize + row] += tanh(outputValue);
    
}

__kernel void forwardPropNoTransTanh_kernel_coarse(
	__global floatType* weight,
	__global floatType* input,
	__global floatType* output,
	int inputLayerSize,
	int outputLayerSize,
	int batchSize
	){
    
	int blockRow = get_group_id(0);
    int blockCol = get_group_id(1);
    
    int outputOffset = outputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * blockRow;
    
    // the output value
    floatType outputValue0 = 0.0;
	floatType outputValue1 = 0.0;
	floatType outputValue2 = 0.0;
	floatType outputValue3 = 0.0;
	floatType outputValue4 = 0.0;
	floatType outputValue5 = 0.0;
	floatType outputValue6 = 0.0;
	floatType outputValue7 = 0.0;
	floatType outputValue8 = 0.0;
	floatType outputValue9 = 0.0;
	floatType outputValue10 = 0.0;
	floatType outputValue11 = 0.0;
	floatType outputValue12 = 0.0;
	floatType outputValue13 = 0.0;
	floatType outputValue14 = 0.0;
	floatType outputValue15 = 0.0;

    
    int row = get_local_id(0);
    
    // main loop
    for(int m = 0; m < (inputLayerSize / BLOCKSIZE); m++){
        
        int weightOffset = outputLayerSize * BLOCKSIZE * m + BLOCKSIZE* blockRow;
        
        int inputOffset = inputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * m;
        
        __local floatType s_input[BLOCKSIZE][BLOCKSIZE];
		
        s_input[row][0] = input[inputOffset + 0 * inputLayerSize + row];
        s_input[row][1] = input[inputOffset + 1 * inputLayerSize + row];
        s_input[row][2] = input[inputOffset + 2 * inputLayerSize + row];
        s_input[row][3] = input[inputOffset + 3 * inputLayerSize + row];
        s_input[row][4] = input[inputOffset + 4 * inputLayerSize + row];
        s_input[row][5] = input[inputOffset + 5 * inputLayerSize + row];
        s_input[row][6] = input[inputOffset + 6 * inputLayerSize + row];
        s_input[row][7] = input[inputOffset + 7 * inputLayerSize + row];
        s_input[row][8] = input[inputOffset + 8 * inputLayerSize + row];
        s_input[row][9] = input[inputOffset + 9 * inputLayerSize + row];
        s_input[row][10] = input[inputOffset + 10 * inputLayerSize + row];
        s_input[row][11] = input[inputOffset + 11 * inputLayerSize + row];
        s_input[row][12] = input[inputOffset + 12 * inputLayerSize + row];
        s_input[row][13] = input[inputOffset + 13 * inputLayerSize + row];
        s_input[row][14] = input[inputOffset + 14 * inputLayerSize + row];
        s_input[row][15] = input[inputOffset + 15 * inputLayerSize + row];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int i = 0; i < BLOCKSIZE; i++){
			floatType i_weight = weight[weightOffset + i * outputLayerSize + row];
            outputValue0 += i_weight * s_input[i][0];
			outputValue1 += i_weight * s_input[i][1];
			outputValue2 += i_weight * s_input[i][2];
			outputValue3 += i_weight * s_input[i][3];
			outputValue4 += i_weight * s_input[i][4];
			outputValue5 += i_weight * s_input[i][5];
			outputValue6 += i_weight * s_input[i][6];
			outputValue7 += i_weight * s_input[i][7];
			outputValue8 += i_weight * s_input[i][8];
			outputValue9 += i_weight * s_input[i][9];
			outputValue10 += i_weight * s_input[i][10];
			outputValue11 += i_weight * s_input[i][11];
			outputValue12 += i_weight * s_input[i][12];
			outputValue13 += i_weight * s_input[i][13];
			outputValue14 += i_weight * s_input[i][14];
			outputValue15 += i_weight * s_input[i][15];
		}

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    output[outputOffset + 0 * outputLayerSize + row] += tanh(outputValue0);
    output[outputOffset + 1 * outputLayerSize + row] += tanh(outputValue1);
    output[outputOffset + 2 * outputLayerSize + row] += tanh(outputValue2);
    output[outputOffset + 3 * outputLayerSize + row] += tanh(outputValue3);
    output[outputOffset + 4 * outputLayerSize + row] += tanh(outputValue4);
    output[outputOffset + 5 * outputLayerSize + row] += tanh(outputValue5);
    output[outputOffset + 6 * outputLayerSize + row] += tanh(outputValue6);
    output[outputOffset + 7 * outputLayerSize + row] += tanh(outputValue7);
    output[outputOffset + 8 * outputLayerSize + row] += tanh(outputValue8);
    output[outputOffset + 9 * outputLayerSize + row] += tanh(outputValue9);
    output[outputOffset + 10 * outputLayerSize + row] += tanh(outputValue10);
    output[outputOffset + 11 * outputLayerSize + row] += tanh(outputValue11);
    output[outputOffset + 12 * outputLayerSize + row] += tanh(outputValue12);
    output[outputOffset + 13 * outputLayerSize + row] += tanh(outputValue13);
    output[outputOffset + 14 * outputLayerSize + row] += tanh(outputValue14);
	output[outputOffset + 15 * outputLayerSize + row] += tanh(outputValue15);
    
    
}

__kernel void forwardPropNoTransLinear_kernel_coarse(
	__global floatType* weight,
	__global floatType* input,
	__global floatType* output,
	int inputLayerSize,
	int outputLayerSize,
	int batchSize
	){
    
	int blockRow = get_group_id(0);
    int blockCol = get_group_id(1);
    
    int outputOffset = outputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * blockRow;
    
    // the output value
    floatType outputValue0 = 0.0;
	floatType outputValue1 = 0.0;
	floatType outputValue2 = 0.0;
	floatType outputValue3 = 0.0;
	floatType outputValue4 = 0.0;
	floatType outputValue5 = 0.0;
	floatType outputValue6 = 0.0;
	floatType outputValue7 = 0.0;
	floatType outputValue8 = 0.0;
	floatType outputValue9 = 0.0;
	floatType outputValue10 = 0.0;
	floatType outputValue11 = 0.0;
	floatType outputValue12 = 0.0;
	floatType outputValue13 = 0.0;
	floatType outputValue14 = 0.0;
	floatType outputValue15 = 0.0;

    
    int row = get_local_id(0);
    
    // main loop
    for(int m = 0; m < (inputLayerSize / BLOCKSIZE); m++){
        
        int weightOffset = outputLayerSize * BLOCKSIZE * m + BLOCKSIZE* blockRow;
        
        int inputOffset = inputLayerSize * BLOCKSIZE * blockCol + BLOCKSIZE * m;
        
        __local floatType s_input[BLOCKSIZE][BLOCKSIZE];
		
        s_input[row][0] = input[inputOffset + 0 * inputLayerSize + row];
        s_input[row][1] = input[inputOffset + 1 * inputLayerSize + row];
        s_input[row][2] = input[inputOffset + 2 * inputLayerSize + row];
        s_input[row][3] = input[inputOffset + 3 * inputLayerSize + row];
        s_input[row][4] = input[inputOffset + 4 * inputLayerSize + row];
        s_input[row][5] = input[inputOffset + 5 * inputLayerSize + row];
        s_input[row][6] = input[inputOffset + 6 * inputLayerSize + row];
        s_input[row][7] = input[inputOffset + 7 * inputLayerSize + row];
        s_input[row][8] = input[inputOffset + 8 * inputLayerSize + row];
        s_input[row][9] = input[inputOffset + 9 * inputLayerSize + row];
        s_input[row][10] = input[inputOffset + 10 * inputLayerSize + row];
        s_input[row][11] = input[inputOffset + 11 * inputLayerSize + row];
        s_input[row][12] = input[inputOffset + 12 * inputLayerSize + row];
        s_input[row][13] = input[inputOffset + 13 * inputLayerSize + row];
        s_input[row][14] = input[inputOffset + 14 * inputLayerSize + row];
        s_input[row][15] = input[inputOffset + 15 * inputLayerSize + row];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int i = 0; i < BLOCKSIZE; i++){
			floatType i_weight = weight[weightOffset + i * outputLayerSize + row];
            outputValue0 += i_weight * s_input[i][0];
			outputValue1 += i_weight * s_input[i][1];
			outputValue2 += i_weight * s_input[i][2];
			outputValue3 += i_weight * s_input[i][3];
			outputValue4 += i_weight * s_input[i][4];
			outputValue5 += i_weight * s_input[i][5];
			outputValue6 += i_weight * s_input[i][6];
			outputValue7 += i_weight * s_input[i][7];
			outputValue8 += i_weight * s_input[i][8];
			outputValue9 += i_weight * s_input[i][9];
			outputValue10 += i_weight * s_input[i][10];
			outputValue11 += i_weight * s_input[i][11];
			outputValue12 += i_weight * s_input[i][12];
			outputValue13 += i_weight * s_input[i][13];
			outputValue14 += i_weight * s_input[i][14];
			outputValue15 += i_weight * s_input[i][15];
		}

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    output[outputOffset + 0 * outputLayerSize + row] += outputValue0;
    output[outputOffset + 1 * outputLayerSize + row] += outputValue1;
    output[outputOffset + 2 * outputLayerSize + row] += outputValue2;
    output[outputOffset + 3 * outputLayerSize + row] += outputValue3;
    output[outputOffset + 4 * outputLayerSize + row] += outputValue4;
    output[outputOffset + 5 * outputLayerSize + row] += outputValue5;
    output[outputOffset + 6 * outputLayerSize + row] += outputValue6;
    output[outputOffset + 7 * outputLayerSize + row] += outputValue7;
    output[outputOffset + 8 * outputLayerSize + row] += outputValue8;
    output[outputOffset + 9 * outputLayerSize + row] += outputValue9;
    output[outputOffset + 10 * outputLayerSize + row] += outputValue10;
    output[outputOffset + 11 * outputLayerSize + row] += outputValue11;
    output[outputOffset + 12 * outputLayerSize + row] += outputValue12;
    output[outputOffset + 13 * outputLayerSize + row] += outputValue13;
    output[outputOffset + 14 * outputLayerSize + row] += outputValue14;
	output[outputOffset + 15 * outputLayerSize + row] += outputValue15;
    
    
}

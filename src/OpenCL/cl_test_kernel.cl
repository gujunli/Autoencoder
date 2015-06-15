// Hello World Kernel

__kernel void hello(__global char* input,
					__global char* output,
					int n){
	// get global position
	int index = get_global_id(0);

	if(index < n){
		output[index] = input[index] + 1;
	}

	return;
}
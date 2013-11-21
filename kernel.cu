#include "kernel.h"


// Copy 1 byte GPU-RAM -> GPU-RAM
__global__
void kernel_gpu_ram(volatile unsigned char * dev_dst_ptr2, volatile unsigned char * dev_src_ptr2) {
	dev_dst_ptr2[0] = dev_src_ptr2[0];
}
// -----------------------------------------------

// Ping-Pong CPU Cores with GPU Cores through pinned mapped CPU-RAM
__global__
void kernel_spin_test(volatile bool * dst_flag_ptr, volatile bool * src_flag_ptr, 
					volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr,
					const bool init_flag, const unsigned int iterations)
{
	bool current_flag = init_flag;

	for(unsigned int i = 0; i < iterations; ++i) 
	{		
		//if(threadIdx.x == 0) 
		{
			while(src_flag_ptr[0] == current_flag);	// spin wait when src_flag_ptr[0] and current_flag will be different

			uva_dst_buff_ptr[0] = uva_src_buff_ptr[0];	// copy 1 byte CPU-RAM -> CPU-RAM
			current_flag = !current_flag;	
		

			dst_flag_ptr[0] = current_flag;			// make that dst_flag_ptr[0] and current_flag will be equal

			__syncthreads();						// wait for all threads
			__threadfence_system();					// sync with CPU-RAM
		}
	}
}
// -----------------------------------------------
// =============================================================
// none-CUDA C++-wrappers
// =============================================================

// Copy 1 byte GPU-RAM -> GPU-RAM
void k_gpu_ram(volatile unsigned char * dev_dst_ptr2, volatile unsigned char * dev_src_ptr2, const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream) 
{
	kernel_gpu_ram<<<BLOCKS, THREADS, 0, stream>>>(dev_dst_ptr2, dev_src_ptr2);
}

// Ping-Pong CPU Cores with GPU Cores through pinned mapped CPU-RAM (Copy 1 bytes CPU-RAM -> CPU-RAM)
void k_spin_test(volatile bool * dst_flag_ptr, volatile bool * src_flag_ptr,
			volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, 
			const bool init_flag, const unsigned int iterations, 
			const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream) 
{
	kernel_spin_test<<<BLOCKS, THREADS, 0, stream>>>(dst_flag_ptr, src_flag_ptr, uva_dst_buff_ptr, uva_src_buff_ptr, 
													init_flag, iterations);
}



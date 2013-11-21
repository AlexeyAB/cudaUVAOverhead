
/// Copy 1 byte GPU-RAM -> GPU-RAM
void k_gpu_ram(volatile unsigned char * dev_dst_ptr2, volatile unsigned char * dev_src_ptr2, 
			   const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream);

/// Ping-Pong CPU Cores with GPU Cores through pinned mapped CPU-RAM (Copy 1 bytes CPU-RAM -> CPU-RAM)
void k_spin_test(volatile bool * dst_flag_ptr, volatile bool * src_flag_ptr,
			volatile unsigned char * uva_dst_buff_ptr, volatile unsigned char * uva_src_buff_ptr, 
			const bool init_flag, const unsigned int iterations,
			const size_t BLOCKS, const size_t THREADS, cudaStream_t &stream);



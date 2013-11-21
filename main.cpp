#include <iostream>
#include <ctime>       // clock_t, clock, CLOCKS_PER_SEC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

// works since CUDA 5.5, Nsight VS 3.1 which works with MSVS 2012 C++11
#if __cplusplus >= 201103L || _MSC_VER >= 1700
	#include <thread>
	using std::thread;
#else
	#include <boost/thread.hpp>
	using boost::thread;
#endif


int main() {

	srand (time(NULL));

	// count devices & info
	int device_count;
	cudaDeviceProp device_prop;

	// get count Cuda Devices
	cudaGetDeviceCount(&device_count);
	std::cout << "Device count: " <<  device_count << std::endl;

	if (device_count > 100) device_count = 0;
	for (int i = 0; i < device_count; i++)
	{
		// get Cuda Devices Info
		cudaGetDeviceProperties(&device_prop, i);
		std::cout << "Device" << i << ": " <<  device_prop.name;
		std::cout << " (" <<  device_prop.totalGlobalMem/(1024*1024) << " MB)";
		std::cout << ", CUDA capability: " <<  device_prop.major << "." << device_prop.minor << std::endl;	
		std::cout << "UVA: " <<  device_prop.unifiedAddressing << std::endl;
		std::cout << "UVA - canMapHostMemory: " <<  device_prop.canMapHostMemory  << std::endl;
		std::cout << "DMA - asyncEngineCount: " <<  device_prop.asyncEngineCount << std::endl;
		std::cout << "MAX BLOCKS NUMBER: " <<  device_prop.maxGridSize[0] << std::endl;
	}
	cudaGetDeviceProperties(&device_prop, 0);
	const size_t dma_controllers_number = device_prop.asyncEngineCount;

	std::cout << std::endl;
	std::cout << "BLOCKS_NUMBER = 1, THREADS_NUMBER = 32" << std::endl;
	//std::cout << "__cplusplus = " << __cplusplus << std::endl;	
	std::cout << std::endl;
	std::cout << "DMA - copy by using DMA-controller (async in each direction)" << std::endl;
	std::cout << "UVA - copy by using Unified Virtual Addressing and launch kernel-function once" << std::endl;
	std::cout << "--------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	// Can Host map memory
	cudaSetDeviceFlags(cudaDeviceMapHost);


	// init pointers

	// src: temp buffer & flag
	unsigned char * host_src_buff_ptr = NULL;
	bool * host_src_flag_ptr = NULL;

	// dst: temp buffer & flag
	unsigned char * host_dst_buff_ptr = NULL;
	bool * host_dst_flag_ptr = NULL;
	
	// temp device memory
	unsigned char * dev_src_ptr = NULL;


	clock_t end, start;

	// sizes of buffer and flag
	static const size_t c_buff_size = 500*1024*1024;
	static const unsigned int c_flags_number = 1;


	// Allocate memory
	cudaHostAlloc(&host_src_buff_ptr, c_buff_size, cudaHostAllocMapped | cudaHostAllocPortable );
	cudaHostAlloc(&host_src_flag_ptr, sizeof(*host_dst_flag_ptr)*c_flags_number, cudaHostAllocMapped | cudaHostAllocPortable );
	for(size_t i = 0; i < c_flags_number; ++i) 
		host_src_flag_ptr[i] = false;

	cudaHostAlloc(&host_dst_buff_ptr, c_buff_size, cudaHostAllocMapped | cudaHostAllocPortable );
	cudaHostAlloc(&host_dst_flag_ptr, sizeof(*host_dst_flag_ptr)*c_flags_number, cudaHostAllocMapped | cudaHostAllocPortable );
	for(size_t i = 0; i < c_flags_number; ++i) 
		host_dst_flag_ptr[i] = false;

	cudaMalloc(&dev_src_ptr, c_buff_size);


	// create none-zero stream
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	

	// =============================================================
	auto func_dma_ping = [&]() -> size_t {
	// Test DMA bandwidth
		static const size_t iters_dma = 40;

		start = clock();
		for(size_t i = 0; i < iters_dma; ++i) {
			cudaMemcpyAsync(dev_src_ptr, host_src_buff_ptr, c_buff_size, cudaMemcpyDefault, stream1);	
			cudaMemcpyAsync(host_src_buff_ptr, dev_src_ptr, c_buff_size, cudaMemcpyDefault, stream2);
		}
		
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

		
		end = clock();

		const float c_time_memcpy = (float)(end - start)/(CLOCKS_PER_SEC);

		const size_t bandwidth = 2*iters_dma*c_buff_size/(1024*1024 * c_time_memcpy);

		std::cout << "DMA Memcpy async: iters = " << iters_dma << 
			", \t time: " << c_time_memcpy << 
			", bandwidth = " << bandwidth << " MB/sec" << std::endl;
		return bandwidth;
	};


	// =============================================================
	auto func_uva_ping = [&]() -> float {
		// Test UVA Ping-Pong
		static const unsigned int iters_uva = 3*1000*1000;

		// src: temp buffer & flag
		unsigned char * uva_src_buff_ptr = NULL;
		bool * uva_src_flag_ptr = NULL;

		// dst: temp buffer & flag
		unsigned char * uva_dst_buff_ptr = NULL;
		bool * uva_dst_flag_ptr = NULL;
		
		// host_ptr -> uva_ptr
		cudaHostGetDevicePointer(&uva_src_buff_ptr, host_src_buff_ptr, 0);
		cudaHostGetDevicePointer(&uva_src_flag_ptr, host_src_flag_ptr, 0);

		cudaHostGetDevicePointer(&uva_dst_buff_ptr, host_dst_buff_ptr, 0);		
		cudaHostGetDevicePointer(&uva_dst_flag_ptr, host_dst_flag_ptr, 0);	



		const bool init_flag = const_cast<volatile bool *>(host_src_flag_ptr)[0] = false;
		//cudaDeviceSynchronize();

		thread t1( [&] () {
			k_spin_test(uva_dst_flag_ptr, uva_src_flag_ptr, uva_dst_buff_ptr, uva_src_buff_ptr, init_flag, iters_uva, 1, 32, stream3);
			cudaStreamSynchronize(stream3);
		} );

		start = clock();
		for(size_t i = 0; i < iters_uva; ++i) {
			//std::cout << i << std::endl;
			// set query flag
			const_cast<volatile bool *>(host_src_flag_ptr)[0] = !const_cast<volatile bool *>(host_src_flag_ptr)[0];
			
			// wait answer flag
			while (const_cast<volatile bool *>(host_dst_flag_ptr)[0] != const_cast<volatile bool *>(host_src_flag_ptr)[0]) {
			}
		}
		end = clock();

		const float c_time_ping_pong = (float)(end - start)/(CLOCKS_PER_SEC);
		const float ping_time = c_time_ping_pong/iters_uva;

		std::cout << "UVA PING-PONG: iters = " << iters_uva << 
			", \t time: " << c_time_ping_pong << 
			", ping = " << ping_time << " sec" << std::endl;
		
		t1.join();
		return ping_time;
	};
	// =============================================================






	cudaDeviceSynchronize();
	// =============================================================
	// TESTS
	// =============================================================
	// Test bandwidth with DMA - alone
	std::cout << "Alone DMA-async: memory copy H->D & D->H: " << std::endl;
	const size_t bandwidth_single = func_dma_ping();
	std::cout << "-----------------------------------------------------------------\n";


	// =============================================================
	// Test ping with UVA - alone
	std::cout << "Alone UVA-ping: 1 byte copy H->D & D->H: " << std::endl;
	func_uva_ping();
	std::cout << "-----------------------------------------------------------------\n";


	// =============================================================
	std::cout << "Overlaped DMA-async and UVA-ping: \n";
	float ping_time;
	// async UVA-Ping-Pong
	thread t2([&]() { ping_time = func_uva_ping();} );
//	thread t2(func_dma_ping);

	// test bandwidth with DMA overlaped with UVA-Ping-Pong
	const size_t bandwidth_overlpaed = func_dma_ping();

	t2.join();
	std::cout << "-----------------------------------------------------------------\n";

	std::cout.precision(3);
	std::cout.width(9);
	std::cout << "Bandwidth for overlaped DMA with UVA slower than alone DMA in: " << 
		100 - 100*(float)bandwidth_overlpaed/(float)bandwidth_single << " %" << std::endl;
	// =============================================================


	// In theory should be performance decrease
	// PCIe
	static const size_t Giga = 1024*1024*1024;
	static const size_t min_size_pcie_TLP = 128;	// 128 bytes could transmit DMA-controller in one package

	// Constants for 16 lane: http://en.wikipedia.org/wiki/PCI_Express
	static const size_t pcie2_bandwitdh = 8*Giga * (8./10.);		// 8GB with encode (8b/10b)
	//static const size_t pcie2_tps = 80*Giga;						// 80 GT/s
	//static const size_t pcie3_bandwitdh = 16*Giga * (128./130.);	// 16GB with encode (128b/130b)
	//static const size_t pcie3_tps = 128*Giga;						// 128 GT/s

	// Summary transactions in one iteration in kernel-function: kernel_spin_test<<<>>>(); in file: kernel.cu
	static const float transactions_per_pingpong_2dir = 6;
	// 2 ops: while(src_flag_ptr[0] == current_flag);
	// 3 ops: uva_dst_buff_ptr[0] = uva_src_buff_ptr[0];
	// 1 ops: dst_flag_ptr[0] = current_flag;
	
	// Average transactions in one direction
	static const float transactions_per_pingpong = transactions_per_pingpong_2dir * dma_controllers_number / 2;

	// transactions per second took
	const float tps_took = transactions_per_pingpong/ping_time;

	// For PCIe 2.0
	//const float percent_tps_took = 100*tps_took/pcie2_tps;
	const float percent_bandwidth_took = 100*tps_took*min_size_pcie_TLP/pcie2_bandwitdh;

	std::cout.precision(3);
	std::cout.width(9);
	std::cout << "In theory should be performance decrease, " <<
		" bandwidth: " << percent_bandwidth_took << " %" << std::endl;

	int b;
	std::cin >> b;

	return 0;
}
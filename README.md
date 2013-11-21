cudaUVAOverhead
===========

Benchmarks of Overhead of using CUDA UVA-access with permanent scan (CUDA 5.5 and C++11)

Source code requires: CUDA >= 4.0 and (C++11 or boost 1.53)

Recomended: CUDA 5.5 and C++11 in MSVS 2012 with Nsight VS 3.1

Comparison bandwidth of async copying by using DMA-controller alone and overlaped with UVA-ping:
- Alone DMA-async (40 iterations X 500MB copy H->D and D->H)
- Alone UVA-ping (3M iterations X 6 transactions to PCIe)
- Overlaped DMA-async and UVA-ping (get decrease of bandwidth of DMA-async copying)

Get decrease of bandwidth of async copying by using DMA-controller when it overlaped and comparison test's results with theory of performance decrease.

Result of benchmarks on GeForce GTX460 SE CC2.1 see in: result.txt
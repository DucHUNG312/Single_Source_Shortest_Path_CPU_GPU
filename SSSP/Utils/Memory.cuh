#pragma once

#include <Utils/Ref.cuh>

#include <pdh.h>
#include <psapi.h>
#include <tchar.h>
#pragma comment(lib, "pdh")

namespace SSSP
{
	template<typename T>
	struct Allocator
	{
		typedef T value_type;

		static T* AllocateHostMemory(std::size_t n)
		{
			if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
				throw std::bad_array_new_length();

			T* allocatedMemory = new T[n];
			if (allocatedMemory != nullptr)
			{
				return allocatedMemory;
			}
			
			throw std::bad_alloc();
		}

		static void DeallocateHostMemory(T* p)
		{
			delete[] p;
		}

		static T* AllocateDeviceMemory(std::size_t n)
		{
			if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
				throw std::bad_array_new_length();

			T* deviceMemory = nullptr;
			CHECK_CUDA_ERROR(cudaMalloc(&deviceMemory, n * sizeof(T)));
			return deviceMemory;
		}

		static void DeallocateDeviceMemory(T* p)
		{
			CHECK_CUDA_ERROR(cudaFree(p));
		}

		static void AllocateMemoryIfNotAllocated(std::size_t n)
		{
			AllocateHostMemory(n);
			AllocateDeviceMemory(n);
		}

		static void CopyHostToDevice(T* dataDevice, T* dataHost, std::size_t n)
		{
			CHECK_CUDA_ERROR(cudaMemcpy(dataDevice, dataHost, n * sizeof(T), cudaMemcpyHostToDevice));
		}

		static void CopyDeviceToHost(T* dataHost, T* dataDevice, std::size_t n)
		{
			CHECK_CUDA_ERROR(cudaMemcpy(dataHost, dataDevice, n * sizeof(T), cudaMemcpyDeviceToHost));
		}
	};

        struct MemoryUse 
        {
            u64 VirtualTotalUsed = 0;
            u64 VirtualProcessUsed = 0;
            u64 VirtualTotalAvailable = 0;
            u64 PhysicalTotalUsed = 0;
            u64 PhysicalProcessUsed = 0;
            u64 PhysicalTotalAvailable = 0;
        };

        struct CPUUse
        {
            f64 ProcessUse = 0.0;
            f64 TotalUse = 0.0;
        };

        class CPUMemMonitor
        {
        public:
            CPUMemMonitor();
            ~CPUMemMonitor();

			CPUUse GetCPUUsage();
			static MemoryUse GetMemoryUsage();

			static CPUMemMonitor* Get() { return s_MemoryMonitor; }
		public:
			HANDLE cpuQuery = NULL;
			HANDLE cpuTotal = NULL;
			ULARGE_INTEGER lastCPU = { 0 };
			ULARGE_INTEGER lastSysCPU = { 0 };
			ULARGE_INTEGER lastUserCPU = { 0 };
			i32 numProcessors = 1;
			HANDLE currentprocess = NULL;
			static CPUMemMonitor* s_MemoryMonitor;
        };
}

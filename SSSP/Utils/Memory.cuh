#pragma once

#include <Core/Core.cuh>
#include <Utils/Ref.cuh>
#include <Utils/Exception.cuh>

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

		static T* AllocateCudaMemory(std::size_t n)
		{
			if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
				throw std::bad_array_new_length();

			T* deviceMemory = nullptr;
			CHECK_CUDA_ERROR(cudaMalloc(&deviceMemory, n * sizeof(T)));
			auto dataDevice = SharedPtr<T>(deviceMemory, [&](T* ptr) { cudaFree(ptr); });
			return dataDevice.get();
		}

		static void AllocateMemoryIfNotAllocated(std::size_t n)
		{
			AllocateHostMemory(n);
			AllocateCudaMemory(n);
		}

		static void CopyHostToDevice(T* dataHost, T* dataDevice, std::size_t n)
		{
			CHECK_CUDA_ERROR(cudaMemcpy(dataDevice, dataHost, n * sizeof(T), cudaMemcpyHostToDevice));
		}

		static void CopyDeviceToHost(T* dataHost, T* dataDevice, std::size_t n)
		{
			CHECK_CUDA_ERROR(cudaMemcpy(dataHost, dataDevice, n * sizeof(T), cudaMemcpyDeviceToHost));
		}
	};
}
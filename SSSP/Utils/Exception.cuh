#include <Core/Core.cuh>

namespace SSSP
{

#define CHECK_CUDA_ERROR(val) CheckCuda( (val), #val, __FILE__, __LINE__ )

	SSSP_FORCE_INLINE void CheckCuda(cudaError_t result, c8 const* const func, const c8* const file, i32 const line)
	{
		if (result != cudaSuccess)
		{
			SSSP_LOG_CRITICAL_NL("Cuda error {}, {}, at {}: {}.{}", result, cudaGetErrorString(result), file, line, func);
			// Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			exit(99);
		}
	}
}
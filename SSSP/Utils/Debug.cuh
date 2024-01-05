#pragma once

#include <Utils/Options.cuh>
#include <Utils/Exception.cuh>

namespace SSSP
{
	namespace Debug
	{
		SSSP_FORCE_INLINE void PrintOptions(const Options& options)
		{
			std::string filePath;
			if (options.dataFile.empty())
			{
				filePath = "Empty";
			}
			SSSP_LOG_DEBUG_NL("[Options] cpu: {}; gpu: {}; hybrid: {}; numThreadOpenMP: {}; filePath: {}", options.cpu, options.gpu, options.hybrid, options.numThreadOpenMP, options.dataFile);
		}

        SSSP_FORCE_INLINE void PrintDist(u32* dist, u32 size) 
        {
            for (i32 i = 0; i < size; i++) 
            {
                SSSP_LOG_DEBUG_NL("dist[{}]: {}", i, dist[i]);
            }
        }

        SSSP_FORCE_INLINE void PrintPreNode(u32* preNode, u32 size) 
        {
            for (i32 i = 0; i < size; i++) 
            {
                SSSP_LOG_DEBUG_NL("prevNode[{}]: {}", i, preNode[i]);
            }
        }

        SSSP_FORCE_INLINE void CompareResult(u32* dist1, u32* dist2, u32 numNodes) 
        {
            u32 diffCount = 0;
            std::vector<i32> nodesId;

            for (i32 i = 0; i < numNodes; i++) 
            {
                if (dist1[i] != dist2[i]) 
                {
                    diffCount++;
                    nodesId.push_back(i);
                }
            }

            if (diffCount == 0) 
            {
                SSSP_LOG_INFO_NL("Good! Short path of each node in the two 2 is the same:");
#ifdef SSSP_DEBUG
                for (i32 i = 0; i < numNodes; i++)
                {
                    SSSP_LOG_TRACE("{}, ", dist1[i]);
                }
#endif // SSSP_DEBUG
            }
            else 
            {
                SSSP_LOG_ERROR_NL("{} of {} does not match: ", diffCount, numNodes);
#ifdef SSSP_DEBUG
                for (i32 i = 0; i < nodesId.size(); i++)
                {
                    SSSP_LOG_ERROR("{}, ", nodesId[i]);
                }
#endif // SSSP_DEBUG
            }
        }

        SSSP_FORCE_INLINE void PrintDeviceInfo()
        {
            i32 nDevices;
            CHECK_CUDA_ERROR(cudaGetDeviceCount(&nDevices));
            for (i32 i = 0; i < nDevices; i++)
            {
                cudaDeviceProp prop;
                CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
                SSSP_LOG_INFO_NL("Device name: {}", prop.name);
                SSSP_LOG_INFO_NL("Memory Clock Rate (KHz): {}", prop.memoryClockRate);
                SSSP_LOG_INFO_NL("Memory Bus Width (bits): {}", prop.memoryBusWidth);
                SSSP_LOG_INFO_NL("Peak Memory Bandwidth (GB/s): {}", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
                SSSP_LOG_INFO_NL();
            }
        }
	}
}
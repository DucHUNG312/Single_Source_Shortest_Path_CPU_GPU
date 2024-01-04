#pragma once

#include <Utils/Options.cuh>

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

        SSSP_FORCE_INLINE void PrintProcessed(bool* processed, u32 size) 
        {
            for (i32 i = 0; i < size; i++) 
            {
                SSSP_LOG_DEBUG_NL("processed[{}]: {}", i, processed[i]);
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
                SSSP_LOG_TRACE_NL("Good! Short path of each node in the two 2 is the same:");
                for (i32 i = 0; i < numNodes; i++) 
                {
                    SSSP_LOG_TRACE("{}, ", dist1[i]);
                }
            }
            else 
            {
                SSSP_LOG_ERROR_NL("{} of {} does not match: ", diffCount, numNodes);
                for (i32 i = 0; i < nodesId.size(); i++) 
                {
                    SSSP_LOG_ERROR("{}, ", nodesId[i]);
                }
                SSSP_LOG_ERROR_NL(" does not match.");
            }
        }
	}
}
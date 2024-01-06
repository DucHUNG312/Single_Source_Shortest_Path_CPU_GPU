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

        SSSP_FORCE_INLINE void PrintPreNode(u32* preNode, u32 size) 
        {
            for (i32 i = 0; i < size; i++) 
            {
                SSSP_LOG_DEBUG_NL("prevNode[{}]: {}", i, preNode[i]);
            }
        }

        SSSP_FORCE_INLINE void CompareResult(u32* dist1, u32* dist2, u32* dist3, u32 numNodes) 
        {
            u32 diffCount = 0;
            u32 diffCount2 = 0;
            std::vector<i32> nodesId;
            std::vector<i32> nodesId2;

            for (i32 i = 0; i < numNodes; i++) 
            {
                if (dist1[i] != dist2[i]) 
                {
                    diffCount++;
                    nodesId.push_back(i);
                }
            }

            for (i32 i = 0; i < numNodes; i++)
            {
                if (dist1[i] != dist3[i])
                {
                    diffCount2++;
                }
            }

            if (diffCount == 0 && diffCount2 == 0)
            {
                SSSP_LOG_INFO_NL("Good! Shortest path of each node in the 3 path are the same:");
#if SSSP_DEBUG
                for (i32 i = 0; i < numNodes; i++)
                {
                    SSSP_LOG_TRACE("{}, ", dist1[i]);
                }
#endif // SSSP_DEBUG
            }
            else 
            {
                SSSP_LOG_ERROR_NL("(dist 1, dist 2): {} does not match: ", diffCount);
                SSSP_LOG_ERROR_NL("(dist 1, dist 3): {} does not match: ", diffCount2);
#if SSSP_DEBUG
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

        SSSP_FORCE_INLINE std::string ToUpperCase(const std::string& str)
        {
            std::string result = str;
            std::locale loc;

            for (char& c : result) 
            {
                c = std::toupper(c, loc);
            }

            return result;
        }

        SSSP_FORCE_INLINE std::string ToPrettyBytes(u64 bytes)
        {
            static auto convlam = [](const auto a_value, const i32 n) 
                {
                std::ostringstream out;
                out << std::fixed << std::setprecision(n) << a_value;
                return out.str();
                };

            const c8* suffixes[7];
            suffixes[0] = " B";
            suffixes[1] = " KB";
            suffixes[2] = " MB";
            suffixes[3] = " GB";
            suffixes[4] = " TB";
            suffixes[5] = " PB";
            suffixes[6] = " EB";
            u32 s = 0; // which suffix to use
            auto count = static_cast<f64>(bytes);
            while (count >= 1024 && s < 7) 
            {
                s++;
                count /= 1024;
            }
            if (count - floor(count) == 0.0)
                return std::to_string((int)count) + suffixes[s];
            else
                return convlam(count, 2) + suffixes[s];
        }
	}
}
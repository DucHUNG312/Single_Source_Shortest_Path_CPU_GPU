#pragma once

#include <Core/Core.cuh>
#include <Utils/Graph.cuh>
#include <Utils/Memory.cuh>
#include <omp.h>

namespace SSSP
{
    SSSP_GLOBAL void UpdateEdgesKernel(i32 numEdges, i32 numEdgesPerThread, u32* dist, u32* preNode, u32* edgesSource, u32* edgesEnd, u32* edgesWeight, bool* finished)
    {
        i32 threadId = blockIdx.x * blockDim.x + threadIdx.x;
        i32 startId = threadId * numEdgesPerThread;

        if (startId >= numEdges)
        {
            return;
        }

        i32 endId = (threadId + 1) * numEdgesPerThread;

        if (endId >= numEdges)
        {
            endId = numEdges;
        }

        for (i32 nodeId = startId; nodeId < endId; nodeId++)
        {
            u32 source = edgesSource[nodeId];
            u32 end = edgesEnd[nodeId];
            u32 weight = edgesWeight[nodeId];

            if (dist[source] + weight < dist[end])
            {
                atomicMin(&dist[end], dist[source] + weight);
                preNode[end] = source;
                *finished = false;
            }
        }
    }

    SSSP_FORCE_INLINE u32* SSSP_GPU_CUDA(SharedPtr<Graph> graph, i32 source, const Options& options)
    {
        i32 numNodes = graph->GetNumNodes();
        i32 numEdges = graph->GetNumEdges();

        // Allocate on CPU
        u32* dist = Allocator<u32>::AllocateHostMemory(numNodes);
        u32* preNode = Allocator<u32>::AllocateHostMemory(numNodes);
        u32* edgesSource = Allocator<u32>::AllocateHostMemory(numEdges);
        u32* edgesEnd = Allocator<u32>::AllocateHostMemory(numEdges);
        u32* edgesWeight = Allocator<u32>::AllocateHostMemory(numEdges);

        i32 numThreads = options.numThreadOpenMP;
        omp_set_num_threads(numThreads);

#pragma omp parallel for shared(dist, preNode) default(none) schedule(static)
        for (i32 i = 0; i < numNodes; i++)
        {
            dist[i] = MAX_DIST;
            preNode[i] = u32(-1);
        }

#pragma omp parallel for shared(dist, preNode, edgesSource, edgesEnd, edgesWeight) default(none) schedule(static)
        for (i32 i = 0; i < numEdges; i++)
        {
            Edge edge = graph->GetEdges().at(i);
            edgesSource[i] = edge.source;
            edgesEnd[i] = edge.end;
            edgesWeight[i] = edge.weight;

            if (edge.source == source && edge.weight < dist[edge.end])
            {
#pragma omp critical
                {
                    if (edge.weight < dist[edge.end])
                    {
                        dist[edge.end] = edge.weight;
                        preNode[edge.end] = source;
                    }
                }
            }
        }
        dist[source] = 0;
        preNode[source] = 0;
        bool finished = true;

        // Allocate on GPU
        u32* d_dist = Allocator<u32>::AllocateDeviceMemory(numNodes);
        u32* d_preNode = Allocator<u32>::AllocateDeviceMemory(numNodes);
        u32* d_edgesSource = Allocator<u32>::AllocateDeviceMemory(numEdges);
        u32* d_edgesEnd = Allocator<u32>::AllocateDeviceMemory(numEdges);
        u32* d_edgesWeight = Allocator<u32>::AllocateDeviceMemory(numEdges);
        bool* d_finished = Allocator<bool>::AllocateDeviceMemory(1);

        // Copy from CPU to GPU
        Allocator<u32>::CopyHostToDevice(d_dist, dist, numNodes);
        Allocator<u32>::CopyHostToDevice(d_preNode, preNode, numNodes);
        Allocator<u32>::CopyHostToDevice(d_edgesSource, edgesSource, numEdges);
        Allocator<u32>::CopyHostToDevice(d_edgesEnd, edgesEnd, numEdges);
        Allocator<u32>::CopyHostToDevice(d_edgesWeight, edgesWeight, numEdges);

        i32 numEdgesPerThread = 8;
        i32 numThreadsPerBlock = 512;
        i32 numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;

        do
        {
            finished = true;
            Allocator<bool>::CopyHostToDevice(d_finished, &finished, 1);

            UpdateEdgesKernel << <numBlock, numThreadsPerBlock >> > (numEdges, numEdgesPerThread, d_dist, d_preNode, d_edgesSource, d_edgesEnd, d_edgesWeight, d_finished);
            CHECK_CUDA_ERROR(cudaPeekAtLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            Allocator<bool>::CopyDeviceToHost(&finished, d_finished, 1);
        } while (!finished);

        // Copy result from GPU to CPU
        Allocator<u32>::CopyDeviceToHost(dist, d_dist, numNodes);

        SSSP_LOG_DEBUG_NL("GPU Process Done!");

        // dealoccate on GPU
        Allocator<u32>::DeallocateDeviceMemory(d_dist);
        Allocator<u32>::DeallocateDeviceMemory(d_preNode);
        Allocator<u32>::DeallocateDeviceMemory(d_edgesSource);
        Allocator<u32>::DeallocateDeviceMemory(d_edgesEnd);
        Allocator<u32>::DeallocateDeviceMemory(d_edgesWeight);
        Allocator<bool>::DeallocateDeviceMemory(d_finished);

        // dealoccate on CPU
        Allocator<u32>::DeallocateHostMemory(preNode);
        Allocator<u32>::DeallocateHostMemory(edgesSource);
        Allocator<u32>::DeallocateHostMemory(edgesEnd);
        Allocator<u32>::DeallocateHostMemory(edgesWeight);

        return dist;
    }
}
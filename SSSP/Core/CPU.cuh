#pragma once

#include <Core/Core.cuh>
#include <Utils/Graph.cuh>
#include <Utils/Memory.cuh>
#include <omp.h>

namespace SSSP
{
    SSSP_FORCE_INLINE u32* SSSP_CPU_Parallel(SharedPtr<Graph> graph, i32 source, const std::string& chromeTracingFile)
    {
        i32 numNodes = graph->GetNumNodes();
        i32 numEdges = graph->GetNumEdges();

        u32* dist = Allocator<u32>::AllocateHostMemory(numNodes);
        u32* preNode = Allocator<u32>::AllocateHostMemory(numNodes);
        u32* edgesSource = Allocator<u32>::AllocateHostMemory(numEdges);
        u32* edgesEnd = Allocator<u32>::AllocateHostMemory(numEdges);
        u32* edgesWeight = Allocator<u32>::AllocateHostMemory(numEdges);

        i32 numThreads = 8;
        omp_set_num_threads(numThreads);

#pragma omp parallel for shared(dist, preNode) default(none) schedule(static)
        for (i32 i = 0; i < numNodes; i++)
        {
            dist[i] = MAX_DIST;
            preNode[i] = u32(-1);
        }

#pragma omp parallel for shared(dist, preNode, edgesSource) default(none) schedule(static)
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
        u32 numIteration = 0;
        bool finished = false;
        bool graphChanged = true;
        while (graphChanged /*!finished*/)
        {
            numIteration++;
            // finished = true;
            graphChanged = false;
#pragma omp parallel
            {
                i32 threadId = omp_get_thread_num();
                //i32 numThreads = omp_get_num_threads();
                i32 numEdgesPerThread = numEdges / numThreads + 1;
                i32 start = threadId * numEdgesPerThread;
                i32 end = (threadId + 1) * numEdgesPerThread;
                if (start > numEdges)
                {
                    start = numEdges;
                }
                if (end > numEdges)
                {
                    end = numEdges;
                }

#pragma omp parallel for shared(dist, preNode) default(none) schedule(static)
                for (i32 i = 0; i < numEdges; i++)
                {        
                    u32 source = edgesSource[i];
                    u32 end = edgesEnd[i];
                    u32 weight = edgesWeight[i];

                    u32 new_dist = dist[source] + weight;

#pragma omp critical
                    {
                        if (new_dist < dist[end])
                        {
                            dist[end] = new_dist;
                            preNode[end] = source;
                            graphChanged = true;
                        }
                    }
                }
            }
        }

        SSSP_LOG_DEBUG_NL("CPU Process Done!");
        SSSP_LOG_DEBUG_NL("Number of Iteration: {}", numIteration);

        return dist;
    } 
}




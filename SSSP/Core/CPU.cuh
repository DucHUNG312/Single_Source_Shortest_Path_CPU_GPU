#pragma once

#include <Core/Core.cuh>
#include <Utils/Graph.cuh>
#include <Utils/Memory.cuh>
#include <omp.h>

namespace SSSP
{
    SSSP_FORCE_INLINE u32* SSSP_CPU_Serial(SharedPtr<Graph> graph, i32 source, const Options& options)
    {
        i32 numNodes = graph->GetNumNodes();
        i32 numEdges = graph->GetNumEdges();

        u32* dist = Allocator<u32>::AllocateHostMemory(numNodes);
        u32* preNode = Allocator<u32>::AllocateHostMemory(numNodes);
        u32* edgesSource = Allocator<u32>::AllocateHostMemory(numEdges);
        u32* edgesEnd = Allocator<u32>::AllocateHostMemory(numEdges);
        u32* edgesWeight = Allocator<u32>::AllocateHostMemory(numEdges);

        for (i32 i = 0; i < numNodes; i++)
        {
            dist[i] = MAX_DIST;
            preNode[i] = u32(-1);
        }

        for (i32 i = 0; i < numEdges; i++) 
        {
            Edge edge = graph->GetEdges().at(i);
            if (edge.source == source) 
            {
                if (edge.weight < dist[edge.end])
                {
                    dist[edge.end] = edge.weight;
                    preNode[edge.end] = source;
                }
            }
            else 
            {
                // Case: edge.source != source
                continue;
            }
        }

        dist[source] = 0;
        preNode[source] = 0;
        bool finished = false;
        //bool graphChanged = true;
        while (!finished)
        {
            finished = true;

            for (i32 i = 0; i < numEdges; i++) 
            {
                Edge edge = graph->GetEdges().at(i);
                // Update its neighbor
                u32 source = edge.source;
                u32 end = edge.end;
                u32 weight = edge.weight;

                if (dist[source] + weight < dist[end]) 
                {
                    dist[end] = dist[source] + weight;
                    preNode[end] = source;
                    //graphChanged = true;
                }
            }
        }

        SSSP_LOG_DEBUG_NL("CPU Serial Process Done!");

        Allocator<u32>::DeallocateHostMemory(preNode);
        Allocator<u32>::DeallocateHostMemory(edgesSource);
        Allocator<u32>::DeallocateHostMemory(edgesEnd);
        Allocator<u32>::DeallocateHostMemory(edgesWeight);

        return dist;
    }

    SSSP_FORCE_INLINE u32* SSSP_CPU_Parallel(SharedPtr<Graph> graph, i32 source, const Options& options)
    {
        i32 numNodes = graph->GetNumNodes();
        i32 numEdges = graph->GetNumEdges();

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
        bool finished = false;
        while (!finished)
        {
            finished = true;
#pragma omp parallel
            {
                i32 threadId = omp_get_thread_num();
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
                    u32 src = edgesSource[i];
                    u32 end = edgesEnd[i];
                    u32 weight = edgesWeight[i];

                    u32 newDist = dist[src] + weight;

                    if (newDist < dist[end])
                    { 
#pragma omp critical
                        {
                            if (newDist < dist[end]) 
                            {
                                dist[end] = newDist;
                                preNode[end] = src;
                            }
                        }
                    }
                }
            }
        }
       
        SSSP_LOG_DEBUG_NL("OpenMP ({} threads) Process Done!", numThreads);

        Allocator<u32>::DeallocateHostMemory(preNode);
        Allocator<u32>::DeallocateHostMemory(edgesSource);
        Allocator<u32>::DeallocateHostMemory(edgesEnd);
        Allocator<u32>::DeallocateHostMemory(edgesWeight);

        return dist;
    } 
}




#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void UpdateEdgesKernelCL(global int* dist, global int* preNode, global int* edgesSource, global int* edgesEnd, global int* edgesWeight, global int* finished, int numEdges)
{
    int threadId = get_global_id(0);

    if (threadId < numEdges)
    {
        int source = edgesSource[threadId];
        int end = edgesEnd[threadId];
        int weight = edgesWeight[threadId];

        if (dist[source] + weight < dist[end])
        {
            dist[end] = dist[source] + weight;
            preNode[end] = source;
            *finished = 0;
        }
    }
}




);} // ############################################################### end of OpenCL C code #####################################################################
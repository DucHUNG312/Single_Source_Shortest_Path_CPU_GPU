__kernel void UpdateEdgesKernelCL(int numEdges, __global int* dist, __global int* preNode, __global int* edgesSource, __global int* edgesEnd, __global int* edgesWeight, __global int* finished) {
    int threadId = get_global_id(0);

    if (threadId < numEdges) {
        int source = edgesSource[threadId];
        int end = edgesEnd[threadId];
        int weight = edgesWeight[threadId];

        if (dist[source] + weight < dist[end]) {
            dist[end] = dist[source] + weight;
            preNode[end] = source;
            *finished = 0;
        }
    }
}
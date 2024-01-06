#include <Utils/Graph.cuh>
#include <omp.h>

namespace SSSP
{
    Graph::Graph(const std::string& graphFilePath)
        : graphFilePath(graphFilePath)
    {
        hasZero = false;
    }

    void Graph::ReadGraph()
    {
		std::ifstream infile;
		infile.open(graphFilePath);

		std::string line;
		std::stringstream ss;
		u32 edgeCounter = 0;
		u32 maxNodeNumber = 0;
		u32 minNodeNumber = MAX_DIST;
		Edge newEdge;
        
        while (getline(infile, line)) 
        {
#pragma omp parallel private(line) reduction(+:edgeCounter) reduction(max:maxNodeNumber) reduction(min:minNodeNumber) reduction(||:hasZero)
            {
#pragma omp for
                for (i32 i = 0; i < 1; i++)
                {
                    // Each thread reads a line
#pragma omp critical
                    {
                        if (!line.empty() && isdigit(line[0]))
                        {
                            ss.clear();
                            ss << line;

                            ss >> newEdge.source; 
                            ss >> newEdge.end;

                            if (ss >> newEdge.weight)
                            {
                                // TODO: load weight
                            }
                            else
                            {
                                newEdge.weight = 1;
                            }

                            hasZero = hasZero || (newEdge.source == 0 || newEdge.end == 0);

                            maxNodeNumber = std::max({ maxNodeNumber, newEdge.source, newEdge.end });
                            minNodeNumber = std::min({ minNodeNumber, newEdge.source, newEdge.end });

                            edges.push_back(newEdge);
                            edgeCounter++;
                        }
                    }
                }
            }
        }

		infile.close();

		if (hasZero) 
		{
			maxNodeNumber++;
		}
		numNodes = maxNodeNumber;
		numEdges = edgeCounter;
		defaultSource = minNodeNumber;

		SSSP_LOG_DEBUG("Read graph from {}. This graph contains {} nodes, and {} edges,", graphFilePath, numNodes, numEdges);
    }

    void Graph::PrintGraph()
    {
		SSSP_LOG_DEBUG_NL("This graph has {} nodes and {} edges.", numNodes, numEdges);
		i32 size = numNodes;

		for (i32 i = 0; i < numEdges; i++) 
		{
			Edge edge = edges.at(i);
			SSSP_LOG_DEBUG_NL("Node {} -> {} ({})", edge.source, edge.end, edge.weight);
		}
    }
}
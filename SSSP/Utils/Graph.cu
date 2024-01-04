#include <Utils/Graph.cuh>

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
			// ignore non graph data
			if (line[0] < '0' || line[0] >'9') 
			{
				continue;
			}

			// stringstream ss(line);
			ss.clear();
			ss << line;
			edgeCounter++;


			ss >> newEdge.source;
			ss >> newEdge.end;

			if (ss >> newEdge.weight) 
			{
				// load weight 
			}
			else 
			{
				// load default weight
				newEdge.weight = 1;
			}

			// graph[start][end] = weight;
			if (newEdge.source == 0) 
			{
				hasZero = true;
			}
			if (newEdge.end == 0) 
			{
				hasZero = true;
			}
			if (maxNodeNumber < newEdge.source) 
			{
				maxNodeNumber = newEdge.source;
			}
			if (maxNodeNumber < newEdge.end) 
			{
				maxNodeNumber = newEdge.end;
			}
			if (minNodeNumber > newEdge.source) 
			{
				minNodeNumber = newEdge.source;
			}
			if (minNodeNumber > newEdge.end) 
			{
				minNodeNumber = newEdge.source;
			}


			edges.push_back(newEdge);

		}

		infile.close();

		if (hasZero) 
		{
			maxNodeNumber++;
		}
		numNodes = maxNodeNumber;
		numEdges = edgeCounter;
		defaultSource = minNodeNumber;

		SSSP_LOG_DEBUG_NL("Read graph from {}. This graph contains {} nodes, and {} edges.", graphFilePath, numNodes, numEdges);
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
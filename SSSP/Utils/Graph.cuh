#pragma once

#include <Core/Core.cuh>
#include <Utils/Options.cuh>

namespace SSSP
{
	struct Edge
	{
		u32 source;
		u32 end;
		u32 weight;
	};

    class Graph
    {
    public:
        Graph(const std::string& graphFilePath);
        void ReadGraph();
        void PrintGraph();

        std::string GetGraphFilePath() { return graphFilePath; }
        u32 GetNumNodes() { return numNodes; }
        u32 GetNumEdges() { return numEdges; }
        u32 GetDefaultSource() { return defaultSource; }
        bool HasZero() { return hasZero; }
        std::vector<Edge> GetEdges() { return edges; }
    private:
        std::string graphFilePath;
        u32 numNodes;
        u32 numEdges;
        u32 defaultSource;
        bool hasZero;
        std::vector<Edge> edges;
    };

}
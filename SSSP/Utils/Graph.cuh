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

    public:
        std::string graphFilePath;
        u32 numNodes;
        u32 numEdges;
        u32 defaultSource = 0;
        bool hasZero;
        std::vector<Edge> edges;
    };

}
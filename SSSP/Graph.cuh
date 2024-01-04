#include <Core/Core.cuh>

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
    private:
        std::string graphFilePath;
        u32 numNodes;
        u32 numEdges;
        u32 defaultSource;
        bool hasZero;
        std::vector<Edge> edges;
    };

}
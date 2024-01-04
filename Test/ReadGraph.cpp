#include <iostream>
#include <string>
#include <sstream>
#include <inttypes.h>
#include <fstream>

using u32 = uint32_t;

struct Edge
{
	u32 source;
	u32 end;
	u32 weight;
};

void ReadGraph(const std::string& graphFilePath)
{
	std::ifstream infile;
	infile.open(graphFilePath);

	std::string line;
	std::stringstream ss;
	u32 edgeCounter = 0;
	u32 maxNodeNumber = 0;
	u32 minNodeNumber = 0;
	bool hasZero = false;

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
	}

	infile.close();

	if (hasZero) 
	{
		maxNodeNumber++;
	}

	std::cout << "Read graph from " << graphFilePath << ". This graph contains " << maxNodeNumber << " nodes, and " << edgeCounter <<" edges." << std::endl;
}

int main()
{
	ReadGraph("E:\\CPUGPU\\DataSets\\data1.txt");
	return 0;
}
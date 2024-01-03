#pragma once

#include <Core/Core.cuh>

namespace SSSP
{
	struct Options
	{
		bool cpu = true;
		bool gpu = true;
		bool hybrid = false;
		i32 numThreadOpenMP = 8;
		std::string dataFile = "E:\\CPUGPU\\DataSets\\simpleGraph.txt"; /// TODO: Set default data path at DataSets folder!
	};
}
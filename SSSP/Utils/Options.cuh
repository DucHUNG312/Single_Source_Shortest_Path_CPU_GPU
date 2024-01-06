#pragma once

#include <Core/Core.cuh>

namespace SSSP
{
	const static std::string s_RootPath = "E:\\CPUGPU\\"; /// TODO: Automate this
	const static std::string chromeTracingFile = s_RootPath + "SSSP\\SSSP.json";

	struct Options
	{
		bool cpu = true;
		bool gpu = true;
		bool hybrid = false;
		i32 numThreadOpenMP = 8;
		std::string dataFile = s_RootPath + "DataSets\\testGraph.txt"; 
		std::string chromeTracingFile = s_RootPath + "SSSP\\SSSP.json";
	};
}
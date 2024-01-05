#include <Core/Core.cuh>
#include <Utils/Args.cuh>
#include <Utils/Graph.cuh>
#include <Core/CPU.cuh>
#include <Core/GPU.cuh>

using namespace SSSP;

#define SSSP_DEBUG 0

const static std::string currentPath = File::GetCurrentExecutablePath();
const static std::string chromeTracingFile = s_RootPath + "SSSP\\SSSP.json";


/// Datasets
const static std::string slashdot0902 = s_RootPath + "DataSets\\Slashdot0902.txt";
const static std::string slashdot0811 = s_RootPath + "DataSets\\Slashdot0811.txt";

i32 main(i32 argc, c8** argv)
{
    /// Init logger (Must to init logger first)
    Logger::Init();

    Options options;
    options = CLIParser::Parse(argc, argv);

    /// Print GPU info
    SSSP_PRINT_DEVICE_STATS;

#if SSSP_DEBUG
    SSSP_PRINT_OPTIONS(options);
#endif // SSSP_DEBUG

    ///
    SSSP_LOG_INFO_NL("*** SIMPLE GRAPH ***\n");

    SSSP_PROFILE_BEGIN_SESSION("Load graphs", chromeTracingFile);
    Graph simpleGraph(options.dataFile);
    SSSP_PROFILE_PRINT_FUNCTION("LoadingSimpleGraph", simpleGraph.ReadGraph());
    SSSP_PROFILE_END_SESSION();

    SSSP_PROFILE_BEGIN_SESSION("Simple Graph", chromeTracingFile);
    //SSSP_PROFILE_PRINT_FUNCTION("SSSP_CPU_Serial", SSSP_CPU_Serial(&simpleGraph, sourceNode, options));
	//SSSP_PROFILE_PRINT_FUNCTION("SSSP_CPU_Parallel", SSSP_CPU_Parallel(&simpleGraph, sourceNode, options));
    //SSSP_PROFILE_PRINT_FUNCTION("SSSP_GPU_CUDA", SSSP_GPU_CUDA(&simpleGraph, sourceNode, options));
    //u32* CPUSerial = SSSP_CPU_Serial(&simpleGraph, sourceNode, options);
    i32 sourceSimpleNode = simpleGraph.defaultSource;
    u32* CPUParallel = SSSP_CPU_Parallel(&simpleGraph, sourceSimpleNode, options);
    u32* GPUCUDA = SSSP_GPU_CUDA(&simpleGraph, sourceSimpleNode, options);
    SSSP_COMPARE_RESULT(CPUParallel, GPUCUDA, simpleGraph.numNodes);
    SSSP_LOG_INFO_NL();
    SSSP_PROFILE_END_SESSION();

    ///
    SSSP_LOG_INFO_NL("*** SLASHDOT0811 GRAPH ***\n");
    SSSP_PROFILE_BEGIN_SESSION("Load graphs", chromeTracingFile);
    Graph slashdot0811_Graph(slashdot0811);
    SSSP_PROFILE_PRINT_FUNCTION("LoadingSlashdot0811", slashdot0811_Graph.ReadGraph());
    SSSP_PROFILE_END_SESSION();

    SSSP_PROFILE_BEGIN_SESSION("Slashdot0811", chromeTracingFile);
    i32 slashdot0811_Node = slashdot0811_Graph.defaultSource;
    u32* slashdot0811_CPUParallel = SSSP_CPU_Parallel(&slashdot0811_Graph, slashdot0811_Node, options);
    u32* slashdot0811_GPUCUDA = SSSP_GPU_CUDA(&slashdot0811_Graph, slashdot0811_Node, options);
    SSSP_COMPARE_RESULT(slashdot0811_CPUParallel, slashdot0811_GPUCUDA, slashdot0811_Graph.numNodes);
    SSSP_LOG_INFO_NL();
    SSSP_PROFILE_END_SESSION();

    ///
    SSSP_LOG_INFO_NL("*** SLASHDOT0902 GRAPH ***\n");
    SSSP_PROFILE_BEGIN_SESSION("Load graphs", chromeTracingFile);
    Graph slashdot0902_Graph(slashdot0902);
    SSSP_PROFILE_PRINT_FUNCTION("LoadingSlashdot0902", slashdot0902_Graph.ReadGraph());
    SSSP_PROFILE_END_SESSION();

    SSSP_PROFILE_BEGIN_SESSION("Slashdot0811", chromeTracingFile);
    i32 slashdot0902_Node = slashdot0902_Graph.defaultSource;
    u32* slashdot0902_CPUParallel = SSSP_CPU_Parallel(&slashdot0902_Graph, slashdot0902_Node, options);
    u32* slashdot0902_GPUCUDA = SSSP_GPU_CUDA(&slashdot0902_Graph, slashdot0902_Node, options);
    SSSP_COMPARE_RESULT(slashdot0902_CPUParallel, slashdot0902_GPUCUDA, slashdot0902_Graph.numNodes);
    SSSP_LOG_INFO_NL();
    SSSP_PROFILE_END_SESSION();

	return 0;
}

#include <Core/Core.cuh>
#include <Utils/Args.cuh>
#include <Utils/Graph.cuh>
#include <Core/CPU.cuh>
#include <Core/GPU.cuh>

using namespace SSSP;

#define SSSP_DEBUG 0

static std::string currentPath = File::GetCurrentExecutablePath();
static std::string chromeTracingFile = currentPath + "\\SSSP.json";

i32 main(i32 argc, c8** argv)
{
    /// Init logger (Must to init logger first)
    Logger::Init();

    /// Print GPU info
    SSSP_PRINT_DEVICE_STATS;

    Options options;
    options = CLIParser::Parse(argc, argv);

    Graph graph(options.dataFile);
    SSSP_PROFILE_PRINT_FUNCTION("LoadingGraph", graph.ReadGraph());

#if SSSP_DEBUG
    SSSP_PRINT_OPTIONS(options);
#endif // SSSP_DEBUG

    i32 sourceNode = graph.GetDefaultSource();

    SSSP_PROFILE_BEGIN_SESSION("Single-Source Shortest Path", chromeTracingFile);

    //SSSP_PROFILE_PRINT_FUNCTION("SSSP_CPU_Serial", SSSP_CPU_Serial(CreateSharedPtr<Graph>(graph), sourceNode, options));
	//SSSP_PROFILE_PRINT_FUNCTION("SSSP_CPU_Parallel", SSSP_CPU_Parallel(CreateSharedPtr<Graph>(graph), sourceNode, options));
    //SSSP_PROFILE_PRINT_FUNCTION("SSSP_GPU_CUDA", SSSP_GPU_CUDA(CreateSharedPtr<Graph>(graph), sourceNode, options));

    //u32* CPUSerial = SSSP_CPU_Serial(CreateSharedPtr<Graph>(graph), sourceNode, options);
    u32* CPUParallel = SSSP_CPU_Parallel(CreateSharedPtr<Graph>(graph), sourceNode, options);
    u32* GPUCUDA = SSSP_GPU_CUDA(CreateSharedPtr<Graph>(graph), sourceNode, options);

    SSSP_COMPARE_RESULT(CPUParallel, GPUCUDA, graph.GetNumNodes());

    SSSP_PROFILE_END_SESSION();

	return 0;
}

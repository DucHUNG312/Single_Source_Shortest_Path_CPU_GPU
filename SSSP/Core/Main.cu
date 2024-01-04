#include <Core/Core.cuh>
#include <Utils/Args.cuh>
#include <Utils/Graph.cuh>
#include <Core/CPU.cuh>

using namespace SSSP;

#define SSSP_DEBUG 0

static std::string currentPath = File::GetCurrentExecutablePath();
static std::string chromeTracingFile = currentPath + "\\SSSP.json";

i32 main(i32 argc, c8** argv)
{
    /// Init logger (Must to init logger first)
    Logger::Init();

    Options options;
    options = CLIParser::Parse(argc, argv);

    Graph graph(options.dataFile);
    SSSP_PROFILE_PRINT_FUNCTION("LoadingGraph", graph.ReadGraph());

#if SSSP_DEBUG
    Debug::PrintOptions(options);
#endif // SSSP_DEBUG

    i32 sourceNode = graph.GetDefaultSource();
	

    SSSP_PROFILE_BEGIN_SESSION("Dataset1", chromeTracingFile);

	SSSP_PROFILE_PRINT_FUNCTION("SSSP_CPU", SSSP_CPU_Parallel(CreateSharedPtr<Graph>(graph), sourceNode, chromeTracingFile));
	
    SSSP_PROFILE_END_SESSION();

	return 0;
}

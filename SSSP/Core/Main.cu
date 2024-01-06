#include <Core/Core.cuh>
#include <Utils/Args.cuh>
#include <Utils/Graph.cuh>
#include <Core/CPU.cuh>
#include <Core/GPU.cuh>

using namespace SSSP;


void RunSSSP(const std::string& dataSetName, const Options& options)
{
    std::string dataPath = s_RootPath + "DataSets\\" + dataSetName + ".txt";
    SSSP_LOG_INFO_NL("*** {} GRAPH ***\n", Debug::ToUpperCase(dataSetName));
    Graph graph(dataPath);
    SSSP_PROFILE_PRINT_FUNCTION(("Load" + dataSetName).c_str(), graph.ReadGraph());
    i32 sourceNode = graph.defaultSource;
    SSSP_LOG_DEBUG_NL();
    u32* CPUSerial = SSSP_CPU_Serial(&graph, sourceNode, options);
    SSSP_LOG_DEBUG_NL();
    u32* CPUParallel = SSSP_CPU_Parallel(&graph, sourceNode, options);
    SSSP_LOG_DEBUG_NL();
    u32* GPUCUDA = SSSP_GPU_CUDA(&graph, sourceNode, options);
    SSSP_COMPARE_RESULT(CPUParallel, GPUCUDA, CPUSerial, graph.numNodes);
    Allocator<u32>::DeallocateHostMemory(GPUCUDA);
    Allocator<u32>::DeallocateHostMemory(CPUSerial);
    Allocator<u32>::DeallocateHostMemory(CPUParallel);
    SSSP_LOG_INFO_NL();
}

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

    SSSP_PROFILE_BEGIN_SESSION("SSSP", chromeTracingFile);

    RunSSSP("simple", options);
    RunSSSP("Gnutella08", options);
    RunSSSP("Gnutella04", options);
    RunSSSP("Gnutella30", options);
    RunSSSP("EmailEuAll", options);
    RunSSSP("Slashdot0811", options);
    RunSSSP("Slashdot0902", options);
    RunSSSP("Amazon0601", options);
    RunSSSP("WikiTalk", options);
    RunSSSP("webGoogle", options);
    RunSSSP("wikiTopcats", options);
    //RunSSSP("socLiveJournal1", options);
    
    SSSP_PROFILE_END_SESSION();

	return 0;
}

#include <Core/Core.cuh>
#include <Utils/Args.cuh>
#include <Graph.cuh>

using namespace SSSP;

#define SSSP_DEBUG 0

void Function1()
{
	SSSP_PROFILE_FUNCTION();

	for (int i = 0; i < 1000; i++)
	{
		SSSP_PROFILE_SCOPE("Function1Loop");
		//std::cout << "Hello World #" << i << std::endl;
	}
}

void Function2()
{
	SSSP_PROFILE_FUNCTION();

	for (int i = 1000; i < 3000; i++)
	{
		SSSP_PROFILE_SCOPE("Function2Loop");
		//std::cout << "Hello World #" << i << std::endl;
	}
}

void RunProfiling()
{
	InstrumentationTimer Timer("RunProfiling");

	Function1();
	Function2();
}

i32 main(i32 argc, c8** argv)
{
    /// Init logger (Must to init logger first)
    Logger::Init();

    Options options;
    options = CLIParser::Parse(argc, argv);

#if SSSP_DEBUG
    Debug::PrintOptions(options);
#endif // SSSP_DEBUG

	static std::string currentPath = File::GetCurrentExecutablePath();
	static std::string chromeTracingFile = currentPath + "\\SSSP.json";

    SSSP_PROFILE_BEGIN_SESSION("Dataset0", chromeTracingFile);
	SSSP_PROFILE_PRINT_FUNCTION("profile", RunProfiling());
	
    SSSP_PROFILE_END_SESSION();

	return 0;
}

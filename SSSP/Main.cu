#include <Core/Core.cuh>
#include <Utils/Args.cuh>
#include <Utils/Debug.cuh>

using namespace SSSP;

#define SSSP_DEBUG 1

void Function1()
{
	SSSP_PROFILE_FUNCTION();

	for (int i = 0; i < 1000; i++)
	{
		SSSP_PROFILE_SCOPE("Function1Loop");
		std::cout << "Hello World #" << i << std::endl;
	}
}

void Function2()
{
	SSSP_PROFILE_FUNCTION();

	for (int i = 1000; i < 3000; i++)
	{
		SSSP_PROFILE_SCOPE("Function2Loop");
		std::cout << "Hello World #" << i << std::endl;
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

	//std::string filepath = File::GetCurrentExecutablePath() + "\\SSSP.json";
    //SSSP_PROFILE_BEGIN_SESSION("Load dataset", filepath);
	//RunProfiling();
    //SSSP_PROFILE_END_SESSION();

	return 0;
}

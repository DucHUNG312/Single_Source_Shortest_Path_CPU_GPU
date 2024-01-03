#include <Utils/Logger.cuh>

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace SSSP
{
	SharedPtr<spdlog::logger> Logger::CoreLogger;
	std::vector<spdlog::sink_ptr> sinks;

	void Logger::Init()
	{
		sinks.emplace_back(CreateSharedPtr<spdlog::sinks::stdout_color_sink_mt>()); // debug
		// sinks.emplace_back(CreateSharedPtr<ImGuiConsoleSink_mt>()); // ImGuiConsole

		auto logFileSink = CreateSharedPtr<spdlog::sinks::rotating_file_sink_mt>("SSSP.log", 1048576 * 5, 3);
		sinks.emplace_back(logFileSink); // Log file
		// create the loggers
		CoreLogger = CreateSharedPtr<spdlog::logger>("SSSPCode", begin(sinks), end(sinks));
		spdlog::register_logger(CoreLogger);

		// configure the loggers
#ifdef LOG_TIMESTAMP
		spdlog::set_pattern("%^[%T] %v%$");
#else
		spdlog::set_pattern("%v%$");
#endif // LOG_TIMESTAMP
		CoreLogger->set_level(spdlog::level::trace);
	}

	void Logger::Release()
	{
		CoreLogger.reset();
		spdlog::shutdown();
	}

	void Logger::AddSink(spdlog::sink_ptr& sink)
	{
		CoreLogger->sinks().push_back(sink);
		CoreLogger->set_pattern("%v%$");
	}
}

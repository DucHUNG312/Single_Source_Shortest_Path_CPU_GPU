#pragma once

#include <Core/Common.cuh>
#include <Utils/Ref.cuh>

#define SPDLOG_EOL ""
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace SSSP
{
	class Logger
	{
	public:
		static void Init();
		static void Release();
		static SharedPtr<spdlog::logger>& GetCoreLogger() { return CoreLogger; }
		static void AddSink(SharedPtr<spdlog::sinks::sink>& sink);
	private:
		static SharedPtr<spdlog::logger> CoreLogger;
	};
}

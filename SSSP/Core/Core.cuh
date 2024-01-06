#pragma once

#include <Core/Common.cuh>

#define SSSP_DEBUG_BUILD
#define SSSP_DEBUG 0

#ifndef SSSP_BUILD_GPU_PATH
#define SSSP_BUILD_GPU_PATH
#endif // !SSSP_BUILD_GPU_PATH

#ifndef SSSP_PLATFORM_WINDOWS
#define SSSP_PLATFORM_WINDOWS
#endif // !SSSP_PLATFORM_WINDOWS


#define SSSP_PROFILE_ENABLED 1

#define MAX_DIST 10e6

// enable log
#include <Utils/Logger.cuh>
// Core log macros
#ifdef SSSP_DEBUG_BUILD
#define SSSP_ENABLE_LOG
#endif // SSSP_DEBUG_BUILD
#ifdef SSSP_ENABLE_LOG
#define SSSP_LOG_TRACE(...) SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::trace, __VA_ARGS__)
#define SSSP_LOG_DEBUG(...) SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::debug, __VA_ARGS__)
#define SSSP_LOG_INFO(...) SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::info, __VA_ARGS__)
#define SSSP_LOG_WARN(...) SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::warn, __VA_ARGS__)
#define SSSP_LOG_ERROR(...) SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::err, __VA_ARGS__)
#define SSSP_LOG_CRITICAL(...) SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::critical, __VA_ARGS__)

#define SSSP_LOG_TRACE_NL(...) (SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::trace, __VA_ARGS__), SSSP_LOG_TRACE('\n'))
#define SSSP_LOG_DEBUG_NL(...) (SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::debug, __VA_ARGS__), SSSP_LOG_TRACE('\n'))
#define SSSP_LOG_INFO_NL(...) (SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::info, __VA_ARGS__), SSSP_LOG_TRACE('\n'))
#define SSSP_LOG_WARN_NL(...) (SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::warn, __VA_ARGS__), SSSP_LOG_TRACE('\n'))
#define SSSP_LOG_ERROR_NL(...) (SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::err, __VA_ARGS__), SSSP_LOG_TRACE('\n'))
#define SSSP_LOG_CRITICAL_NL(...) (SPDLOG_LOGGER_CALL(::SSSP::Logger::GetCoreLogger(), spdlog::level::level_enum::critical, __VA_ARGS__), SSSP_LOG_TRACE('\n'))
#else
#define SSSP_LOG_TRACE(...) ((void)0)
#define SSSP_LOG_DEBUG(...) ((void)0)
#define SSSP_LOG_INFO(...) ((void)0)
#define SSSP_LOG_WARN(...) ((void)0)
#define SSSP_LOG_ERROR(...) ((void)0)
#define SSSP_LOG_CRITICAL(...) ((void)0)

#define SSSP_LOG_TRACE_NL(...) ((void)0)
#define SSSP_LOG_DEBUG_NL(...) ((void)0)
#define SSSP_LOG_INFO_NL(...) ((void)0)
#define SSSP_LOG_WARN_NL(...) ((void)0)
#define SSSP_LOG_ERROR_NL(...) ((void)0)
#define SSSP_LOG_CRITICAL_NL(...) ((void)0)
#endif


#ifdef SSSP_PLATFORM_WINDOWS
#pragma warning(disable : 4251)
#ifdef SSSP_DYNAMIC
#ifdef SSSP_ENGINE
#define SSSP_EXPORT __declspec(dllexport)
#else
#define SSSP_EXPORT __declspec(dllimport)
#endif
#else
#define SSSP_EXPORT
#endif
#define SSSP_HIDDEN
#else
#define SSSP_EXPORT __attribute__((visibility("default")))
#define SSSP_HIDDEN __attribute__((visibility("hidden")))
#endif

#ifdef SSSP_PLATFORM_WINDOWS
#define SSSP_BREAK() __debugbreak()
#else
#define SSSP_BREAK() raise(SIGTRAP)
#endif

#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS))
#define SSSP_EXCEPTIONS
#endif

#if defined(_MSC_VER)
#define SSSP_IS_MSCV
#define DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISABLE_WARNING_POP __pragma(warning(pop))
#define DISABLE_WARNING(warningNumber) __pragma(warning(disable \
                                                        : warningNumber))
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(4505)
#define DISABLE_WARNING_CONVERSION_TO_SMALLER_TYPE DISABLE_WARNING(4267)
#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
#define DISABLE_WARNING_CONVERSION_TO_SMALLER_TYPE
#endif

#ifndef SSSP_FORCE_INLINE
#if defined(_MSC_VER)
#define SSSP_FORCE_INLINE __forceinline
#else
#define SSSP_FORCE_INLINE inline
#endif
#endif // !SSSP_FORCE_INLINE

#ifndef SSSP_ALWAYS_INLINE
#if defined(_MSC_VER)
#define SSSP_ALWAYS_INLINE SSSP_FORCE_INLINE
#else
#define SSSP_ALWAYS_INLINE inline
#endif 
#endif // !SSSP_ALWAYS_INLINE

#ifndef SSSP_NODISCARD
#define SSSP_NODISCARD [[nodiscard]]
#endif // !SSSP_NODISCARD

#ifndef SSSP_ALLOW_DISCARD
#define SSSP_ALLOW_DISCARD (void)
#endif // !SSSP_ALLOW_DISCARD

#if defined(_MSC_VER)
  // NOTE MSVC often gives C4127 warnings with compiletime if statements. See bug 1362.
  // This workaround is ugly, but it does the job.
#define SSSP_CONST_CONDITIONAL(cond)  (void)0, cond
#else
#define SSSP_CONST_CONDITIONAL(cond)  cond
#endif

// GPU Stuff
#if defined(SSSP_BUILD_GPU_PATH)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#define SSSP_IS_GPU_CODE
#ifndef SSSP_NOINLINE
#define SSSP_NOINLINE __attribute__((noinline))
#endif
#define SSSP_CPU_GPU __host__ __device__
#define SSSP_CPU __host__
#define SSSP_GPU __device__
#define SSSP_GLOBAL __global__
#if defined(SSSP_IS_GPU_CODE)
#define SSSP_CONST __device__ const
#else
#define SSSP_CONST const
#endif
#else
#define SSSP_CONST const
#define SSSP_CPU_GPU
#define SSSP_GPU
#define SSSP_CPU
#define SSSP_GLOBAL
#endif

#ifdef SSSP_IS_WINDOWS
#define SSSP_CPU_GPU_LAMBDA(...) [ =, *this ] SSSP_CPU_GPU(__VA_ARGS__) mutable
#else
#define SSSP_CPU_GPU_LAMBDA(...) [=] SSSP_CPU_GPU(__VA_ARGS__)
#endif

// Define Cache Line Size Constant
#ifdef SSSP_BUILD_GPU_PATH
#define SSSP_L1_CACHE_LINE_SIZE 128
#else
#define SSSP_L1_CACHE_LINE_SIZE 64
#endif

// enable check
#define SSSP_ENABLE_CHECK
#ifdef SSSP_ENABLE_CHECK
#define CHECK(x) (!(!(x) && (SSSP_LOG_CRITICAL("Check failed: {}", #x), SSSP_BREAK(), true)))
#define CHECK_MSG(x, msg) (!(!(x) && (SSSP_LOG_ERROR(#msg), true)))
#define CHECK_IMPL(a, b, op) (!(!(a op b) && (SSSP_LOG_CRITICAL("Check failed: {}{}{}", #a, #op, #b), SSSP_BREAK(), true)))
#define EMPTY_CHECK \
	do {			\
	} while (false);
#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

#define SSSP_CHECK(x) (CHECK(x))
#define SSSP_CHECK_MSG(x, msg) (CHECK_MSG(x, msg))
#define SSSP_CHECK_EQ(a, b) (CHECK_EQ(a, b))
#define SSSP_CHECK_NE(a, b) (CHECK_NE(a, b))
#define SSSP_CHECK_GT(a, b) (CHECK_GT(a, b))
#define SSSP_CHECK_GE(a, b) (CHECK_GE(a, b))
#define SSSP_CHECK_LT(a, b) (CHECK_LT(a, b))
#define SSSP_CHECK_LE(a, b) (CHECK_LE(a, b))
#else
#define SSSP_CHECK(x) EMPTY_CHECK
#define SSSP_CHECK_MSG(x, msg) EMPTY_CHECK
#define SSSP_CHECK_MSG_RET(x, ret, msg) EMPTY_CHECK
#define SSSP_CHECK_EQ(a, b) EMPTY_CHECK
#define SSSP_CHECK_NE(a, b) EMPTY_CHECK
#define SSSP_CHECK_GT(a, b) EMPTY_CHECK
#define SSSP_CHECK_GE(a, b) EMPTY_CHECK
#define SSSP_CHECK_LT(a, b) EMPTY_CHECK
#define SSSP_CHECK_LE(a, b) EMPTY_CHECK
#endif // SSSP_ENABLE_CHECK

#ifndef SSSP_STR
#define SSSP_STR(x) #x
#define SSSP_MAKE_STR(x) STR(x)
#endif // !SSSP_STR

#ifndef SSSP_MAX_RECURSION
#define SSSP_MAX_RECURSION 100
#endif // SSSP_MAX_RECURSION

#ifndef BIT
#define BIT(x) (1 << x)
#endif // !BIT

#ifndef SSSP_SHIFT_LEFT
#define SSSP_SHIFT_LEFT(x) (std::size(1) << x)
#endif // 

#ifndef SSSP_SHIFT_RIGHT
#define SSSP_SHIFT_RIGHT(x) (std::size(1) >> x)
#endif // !SSSP_SHIFT_RIGHT

#ifndef SSSP_CONCAT
#define SSSP_CONCAT_HELPER(x, y) x##y
#define SSSP_CONCAT(x, y) SSSP_CONCAT_HELPER(x, y)
#endif // SSSP_SSSP_CONCAT

#ifndef SSSP_MEM_LIMIT_MB
#define SSSP_MEM_LIMIT_MB 32
#endif

namespace SSSP
{
    SSSP_FORCE_INLINE void CheckCuda(cudaError_t result, c8 const* const func, const c8* const file, i32 const line)
    {
        if (result != cudaSuccess)
        {
            SSSP_LOG_CRITICAL_NL("Cuda error {}, {}, at {}: {}.{}", result, cudaGetErrorString(result), file, line, func);
            // Make sure we call CUDA Device Reset before exiting
            cudaDeviceReset();
            exit(99);
        }
    }
}

#define CHECK_CUDA_ERROR(val) ::SSSP::CheckCuda( (val), #val, __FILE__, __LINE__ )

#if SSSP_PROFILE_ENABLED
#include <Utils/Instrumentor.cuh>
#include <Utils/Debug.cuh>
#define EMPTY_PROFILE \
	do {			\
	} while (false);
#if defined(__FUNCSIG__)
#define SSSP_PROFILE_FUNC_SIG __FUNCSIG__
#elif defined(__cplusplus)
#define SSSP_PROFILE_FUNC_SIG __func__
#else
#define SSSP_PROFILE_FUNC_SIG "SSSP_PROFILE_FUNC_SIG unknown!"
#endif
#define SSSP_PROFILE_BEGIN_SESSION(name, filepath) ::SSSP::Instrumentor::Get().BeginSession(name, filepath)
#define SSSP_PROFILE_END_SESSION() ::SSSP::Instrumentor::Get().EndSession()
#define SSSP_PROFILE_SCOPE(name) ::SSSP::InstrumentationTimer timer##__LINE__(name)
#define SSSP_PROFILE_FUNCTION()  SSSP_PROFILE_SCOPE(SSSP_PROFILE_FUNC_SIG)
#define SSSP_PROFILE_PRINT_FUNCTION(name, func)                                                                                     \
    do {                                                                                                                            \
        static i32 unique_counter_##__COUNTER__ = 0;                                                                                \
        ::SSSP::InstrumentationTimer timer_##__COUNTER__(name);                                                                     \
        unique_counter_##__COUNTER__++;                                                                                             \
        func;                                                                                                                       \
    } while(0)                           

#else
#define SSSP_PROFILE_BEGIN_SESSION(name, filepath) EMPTY_PROFILE
#define SSSP_PROFILE_END_SESSION() EMPTY_PROFILE
#define SSSP_PROFILE_SCOPE(name) EMPTY_PROFILE
SSSP_PROFILE_PRINT_FUNCTION(name, func) EMPTY_PROFILE
#define SSSP_PROFILE_FUNCTION() EMPTY_PROFILE
#endif

// Debug
#define SSSP_PRINT_DEVICE_STATS ::SSSP::Debug::PrintDeviceInfo()
#define SSSP_PRINT_OPTIONS(options) ::SSSP::Debug::PrintOptions(options)
#define SSSP_PRINT_DIST(dist, size) ::SSSP::Debug::PrintDist(dist, size)
#define SSSP_PRINT_PRENODE(preNode, size) ::SSSP::Debug::PrintPreNode(preNode, size)
#define SSSP_COMPARE_RESULT(dist1, dist2, dist3, numNodes) ::SSSP::Debug::CompareResult(dist1, dist2, dist3, numNodes)

// Math utils
#include <Utils/Numeric.cuh>
#define SSSP_IS_NAN(x) IsNaN(x)
#define SSSP_IS_INF(x) IsInf(x)
#define SSSP_IS_FINITE(x) IsFinite(x)
#define SSSP_IS_ZERO(x) ((x) == 0)
#define SSSP_IS_VALID_NUM(x) IsValidNum(x)
#define SSSP_IS_VALID_OP(a, b, op) (!(!SSSP_IS_VALID_NUM(a op b)) && (SSSP_LOG_CRITICAL("Check failed: {}{}{}", #a, #op, #b), SSSP_BREAK(), true))
#define SSSP_IS_VALID_ADD(a, b) SSSP_IS_VALID_OP(a, b, +)
#define SSSP_IS_VALID_SUB(a, b) SSSP_IS_VALID_OP(a, b, -)
#define SSSP_IS_VALID_MUL(a, b) SSSP_IS_VALID_OP(a, b, *)
#define SSSP_IS_VALID_DIV(a, b) SSSP_IS_VALID_OP(a, b, /)


#define SSSP_UNIMPLEMENTED														\
    {																		\
        SSSP_LOG_CRITICAL("Unimplemented : {} : {}", __FILE__, __LINE__); \
        SSSP_BREAK();														\
    }

#define SSSP_NONCOPYABLE(class_name)							  \
    class_name(const class_name&)            = delete;		  \
    class_name& operator=(const class_name&) = delete;
#define SSSP_NONCOPYABLEANDMOVE(class_name)					  \
    class_name(const class_name&)            = delete;		  \
    class_name& operator=(const class_name&) = delete;		  \
    class_name(class_name&&)                 = delete;		  \
    class_name& operator=(class_name&&)      = delete;

#define SSSP_CANCOPYABLE(class_name)							  \
    class_name(const class_name&)            = default;		  \
    class_name& operator=(const class_name&) = default;
#define SSSP_CANCOPYABLEANDMOVE(class_name)					  \
    class_name(const class_name&)            = default;		  \
    class_name& operator=(const class_name&) = default;		  \
    class_name(class_name&&)                 = default;		  \
    class_name& operator=(class_name&&)      = default;

// Suppresses 'unused variable' warnings.
namespace SSSP
{
    namespace Internal
    {
        template<typename T>
        SSSP_CPU_GPU SSSP_FORCE_INLINE
            void ignore_unused_variable(const T&) {}

        template <typename T, uint64_t N>
        auto ArraySizeHelper(const T(&array)[N]) -> char(&)[N];
    }
}
#define SSSP_UNUSED_VARIABLE(var) ::SSSP::Internal::ignore_unused_variable(var);
#define SSSP_ARRAYSIZE(array) (sizeof(::SSSP::Internal::ArraySizeHelper(array)))

#include <Utils/Memory.cuh>
#ifndef SSSP_MEMORY_TRACKING
#define SSSP_MEMORY_TRACKING                                                                                                        \
auto memusage = ::SSSP::CPUMemMonitor::GetMemoryUsage();                                                                            \
SSSP_LOG_DEBUG_NL("Physical Process Memory Usage: {}", ::SSSP::Debug::ToPrettyBytes(memusage.PhysicalProcessUsed));                 \
SSSP_LOG_DEBUG_NL("Virtual Process Memory Usage: {}",  ::SSSP::Debug::ToPrettyBytes(memusage.VirtualProcessUsed))
#endif // !SSSP_MEMORY_TRACKING




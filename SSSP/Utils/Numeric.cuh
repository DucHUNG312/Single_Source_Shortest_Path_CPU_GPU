#pragma once

#include <Utils/TypeTraits.cuh>

namespace SSSP
{
#ifdef SSSP_IS_GPU_CODE

#define DoubleOneMinusEpsilon 0x1.fffffffffffffp-1
#define FloatOneMinusEpsilon float(0x1.fffffep-1)

#ifdef SSSP_FLOAT_AS_DOUBLE
#define OneMinusEpsilon DoubleOneMinusEpsilon
#else
#define OneMinusEpsilon FloatOneMinusEpsilon
#endif

#define Infinity std::numeric_limits<f32>::infinity()
#define MachineEpsilon std::numeric_limits<f32>::epsilon() * 0.5f
#else
	// Floating-point Constants
	static constexpr f32 Infinity = std::numeric_limits<f32>::infinity();
	static constexpr f32 MachineEpsilon = std::numeric_limits<f32>::epsilon() * 0.5;
	static constexpr double DoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
	static constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;
#ifdef SSSP_FLOAT_AS_DOUBLE
	static constexpr double OneMinusEpsilon = DoubleOneMinusEpsilon;
#else
	static constexpr float OneMinusEpsilon = FloatOneMinusEpsilon;
#endif
#endif  // SSSP_IS_GPU_CODE

	template<typename T>
	SSSP_FORCE_INLINE SSSP_CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool> IsNaN(T v)
	{
#ifdef SSSP_IS_GPU_CODE
		return isnan(v);
#else
		return std::isnan(v);
#endif // SSSP_IS_GPU_CODE
	}

	template<typename T>
	SSSP_FORCE_INLINE SSSP_CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, bool> IsNaN(T v)
	{
		return false;
	}

	template<typename T>
	SSSP_FORCE_INLINE SSSP_CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool> IsInf(T v)
	{
#ifdef SSSP_IS_GPU_CODE
		return isinf(v);
#else
		return std::isinf(v);
#endif // SSSP_IS_GPU_CODE
	}

	template<typename T>
	SSSP_FORCE_INLINE SSSP_CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, bool> IsInf(T v)
	{
		return false;
	}

	template<typename T>
	SSSP_FORCE_INLINE SSSP_CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool> IsFinite(T v)
	{
#ifdef SSSP_IS_GPU_CODE
		return isfinite(v);
#else
		return std::isfinite(v);
#endif // SSSP_IS_GPU_CODE
	}

	template<typename T>
	SSSP_FORCE_INLINE SSSP_CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, bool> IsFinite(T v)
	{
		return true;
	}

	template<typename T>
	SSSP_FORCE_INLINE SSSP_CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool> IsValidNum(T v)
	{
#ifdef SSSP_IS_GPU_CODE
		return !isnan(v) && !isinf(v) && (SSSP_CHECK_LT(v, std::numeric_limits<T>::max()) && SSSP_CHECK_GT(v, std::numeric_limits<T>::min()));
#else
		return !std::isnan(v) && !std::isinf(v) && (SSSP_CHECK_LT(v, std::numeric_limits<T>::max()) && SSSP_CHECK_GT(v, std::numeric_limits<T>::min()));
#endif // SSSP_IS_GPU_CODE
	}
}


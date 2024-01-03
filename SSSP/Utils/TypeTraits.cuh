#pragma once

#include <Core/Common.cuh>

namespace SSSP
{
	template<typename>
	struct Uint
	{
		using type = u32;
	};

	template<>
	struct Uint<u8>
	{
		using type = u8;
	};

	template<>
	struct Uint<u16>
	{
		using type = u16;
	};

	template<>
	struct Uint<u64>
	{
		using type = u64;
	};

	template<typename>
	struct Int
	{
		using type = i32;
	};

	template<>
	struct Int<i8>
	{
		using type = i8;
	};

	template<>
	struct Int<i16>
	{
		using type = i16;
	};

	template<>
	struct Int<i64>
	{
		using type = i64;
	};

	template<typename>
	struct Float
	{
		using type = f32;
	};

	template<>
	struct Float<f64>
	{
		using type = f64;
	};

	template<>
	struct Float<f128>
	{
		using type = f128;
	};
}

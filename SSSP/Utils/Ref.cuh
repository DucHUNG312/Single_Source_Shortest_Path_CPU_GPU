#pragma once

#include <Core/Common.cuh>

namespace SSSP
{
	//#define CUSTOM_SMART_PTR
#ifdef CUSTOM_SMART_PTR

#else
	template <class T>
	using SharedPtr = std::shared_ptr<T>;

	template <typename T, typename... Args>
	SharedPtr<T> CreateSharedPtr(Args&&... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}

	template <class T>
	using WeakPtr = std::weak_ptr<T>;

	template <class T>
	using UniquePtr = std::unique_ptr<T>;

	template <typename T, typename... Args>
	UniquePtr<T> CreateUniquePtr(Args&&... args)
	{
		return std::make_unique<T>(std::forward<Args>(args)...);
	}
#endif
}

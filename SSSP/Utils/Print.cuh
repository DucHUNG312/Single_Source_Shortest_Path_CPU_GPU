#pragma once

#include <Core/Core.cuh>
#include <Utils/Ref.cuh>


namespace SSSP
{
	template <typename... Args>
	SSSP_FORCE_INLINE std::string StringPrintf(const c8* fmt, Args &&...args);
}

namespace SSSP
{
	// helpers, fwiw
	template <typename T>
	static auto operator<<(std::ostream& os, const T& v) -> decltype(v.ToString(), os)
	{
		return os << v.ToString();
	}
	template <typename T>
	static auto operator<<(std::ostream& os, const T& v) -> decltype(ToString(v), os)
	{
		return os << ToString(v);
	}

	template <typename T>
	SSSP_FORCE_INLINE std::ostream& operator<<(std::ostream& os, const SharedPtr<T>& p)
	{
		if (p)
			return os << p->ToString();
		else
			return os << "(nullptr)";
	}

	template <typename T>
	SSSP_FORCE_INLINE std::ostream& operator<<(std::ostream& os, const UniquePtr<T>& p)
	{
		if (p)
			return os << p->ToString();
		else
			return os << "(nullptr)";
	}

	namespace Internal
	{

		std::string FloatToString(f32 v);
		std::string DoubleToString(f64 v);

		template <typename T>
		struct IntegerFormatTrait;

		template <>
		struct IntegerFormatTrait<bool>
		{
			static constexpr const c8* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<c8>
		{
			static constexpr const c8* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<uc8>
		{
			static constexpr const c8* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<i32>
		{
			static constexpr const c8* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<u32>
		{
			static constexpr const c8* fmt() { return "u"; }
		};
		template <>
		struct IntegerFormatTrait<i16>
		{
			static constexpr const c8* fmt() { return "d"; }
		};
		template <>
		struct IntegerFormatTrait<u16>
		{
			static constexpr const c8* fmt() { return "u"; }
		};
		template <>
		struct IntegerFormatTrait<long>
		{
			static constexpr const c8* fmt() { return "ld"; }
		};
		template <>
		struct IntegerFormatTrait<unsigned long>
		{
			static constexpr const c8* fmt() { return "lu"; }
		};
		template <>
		struct IntegerFormatTrait<i64>
		{
			static constexpr const c8* fmt() { return "lld"; }
		};
		template <>
		struct IntegerFormatTrait<u64>
		{
			static constexpr const c8* fmt() { return "llu"; }
		};

		template <typename T>
		using HasSize =
			std::is_integral<typename std::decay_t<decltype(std::declval<T&>().size())>>;

		template <typename T>
		using HasData =
			std::is_pointer<typename std::decay_t<decltype(std::declval<T&>().data())>>;

		// Don't use size()/data()-based operator<< for std::string...
		SSSP_FORCE_INLINE std::ostream& operator<<(std::ostream& os, const std::string& str) {
			return std::operator<<(os, str);
		}

		template <typename T>
		SSSP_FORCE_INLINE std::enable_if_t<HasSize<T>::value&& HasData<T>::value, std::ostream&>
			operator<<(std::ostream& os, const T& v)
		{
			os << "[ ";
			auto ptr = v.data();
			for (size_t i = 0; i < v.size(); ++i)
			{
				os << ptr[i];
				if (i < v.size() - 1)
					os << ", ";
			}
			return os << " ]";
		}

		// base case
		void StringPrintfRecursive(std::string* s, const c8* fmt);

		// 1. Copy from fmt to *s, up to the next formatting directive.
		// 2. Advance fmt past the next formatting directive and return the
		//    formatting directive as a string.
		std::string CopyToFormatString(const c8** fmt_ptr, std::string* s);

		template <typename T>
		SSSP_FORCE_INLINE typename std::enable_if_t<!std::is_class_v<typename std::decay_t<T>>, std::string>
			FormatOne(const c8* fmt, T&& v) {
			// Figure out how much space we need to allocate; add an extra
			// c8acter for the '\0'.
			size_t size = snprintf(nullptr, 0, fmt, v) + 1;
			std::string str;
			str.resize(size);
			snprintf(&str[0], size, fmt, v);
			str.pop_back();  // remove trailing NUL
			return str;
		}

		template <typename T>
		SSSP_FORCE_INLINE typename std::enable_if_t<std::is_class_v<typename std::decay_t<T>>, std::string>
			FormatOne(const c8* fmt, T&& v)
		{
			SSSP_LOG_CRITICAL_NL("Printf: Non-basic type %s passed for format string %s", typeid(v).name(), fmt);
			return "";
		}

		template <typename T, typename... Args>
		SSSP_FORCE_INLINE void StringPrintfRecursive(std::string* s, const c8* fmt, T&& v, Args &&...args);

		template <typename T, typename... Args>
		SSSP_FORCE_INLINE void StringPrintfRecursiveWithPrecision(std::string* s, const c8* fmt, const std::string& nextFmt, T&& v, Args &&...args)
		{
			SSSP_LOG_CRITICAL_NL("MEH");
		}

		template <typename T, typename... Args>
		SSSP_FORCE_INLINE typename std::enable_if_t<!std::is_class_v<typename std::decay_t<T>>, void>
			StringPrintfRecursiveWithPrecision(std::string* s, const c8* fmt, const std::string& nextFmt, i32 precision, T&& v, Args &&...args)
		{
			size_t size = snprintf(nullptr, 0, nextFmt.c_str(), precision, v) + 1;
			std::string str;
			str.resize(size);
			snprintf(&str[0], size, nextFmt.c_str(), precision, v);
			str.pop_back();  // remove trailing NUL
			*s += str;
			StringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
		}

#ifdef SSSP_IS_MSCV
#pragma warning(push)
#pragma warning(disable : 4102)  // bogus "unreferenced label" warning for done: below
#endif

		// General-purpose version of StringPrintfRecursive; add the formatted
		// output for a single StringPrintf() argument to the final result string
		// in *s.
		template <typename T, typename... Args>
		SSSP_FORCE_INLINE void StringPrintfRecursive(std::string* s, const c8* fmt, T&& v, Args &&...args)
		{
			std::string nextFmt = CopyToFormatString(&fmt, s);
			bool precisionViaArg = nextFmt.find('*') != std::string::npos;

			bool isSFmt = nextFmt.find('s') != std::string::npos;
			bool isDFmt = nextFmt.find('d') != std::string::npos;

			if constexpr (std::is_integral_v<std::decay_t<T>>)
			{
				if (precisionViaArg)
				{
					StringPrintfRecursiveWithPrecision(s, fmt, nextFmt, v,
						std::forward<Args>(args)...);
					return;
				}
			}
			else if (precisionViaArg)
				SSSP_LOG_CRITICAL_NL("Non-integral type provided for %* format.");

			if constexpr (std::is_same_v<std::decay_t<T>, f32>)
				if (nextFmt == "%f" || nextFmt == "%s")
				{
					*s += Internal::FloatToString(v);
					goto done;
				}

			if constexpr (std::is_same_v<std::decay_t<T>, f64>)
				if (nextFmt == "%f" || nextFmt == "%s")
				{
					*s += Internal::DoubleToString(v);
					goto done;
				}

			if constexpr (std::is_same_v<std::decay_t<T>, bool>)  // FIXME: %-10s with bool
				if (isSFmt)
				{
					*s += bool(v) ? "true" : "false";
					goto done;
				}

			if constexpr (std::is_integral_v<std::decay_t<T>>)
			{
				if (isDFmt)
				{
					nextFmt.replace(nextFmt.find('d'), 1,
						Internal::IntegerFormatTrait<std::decay_t<T>>::fmt());
					*s += FormatOne(nextFmt.c_str(), std::forward<T>(v));
					goto done;
				}
			}
			else if (isDFmt)
				SSSP_LOG_CRITICAL_NL("Non-integral type passed to %d format.");

			if (isSFmt)
			{
				std::stringstream ss;
				ss << v;
				*s += FormatOne(nextFmt.c_str(), ss.str().c_str());
			}
			else if (!nextFmt.empty())
				*s += FormatOne(nextFmt.c_str(), std::forward<T>(v));
			else
				SSSP_LOG_CRITICAL_NL("Excess values passed to Printf.");
		done:
			StringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
		}

#ifdef SSSP_IS_MSCV
#pragma warning(pop)
#endif

	}  // namespace detail

	// Printing Function Declarations
	template <typename... Args>
	void Printf(const c8* fmt, Args &&...args);
	template <typename... Args>
	SSSP_FORCE_INLINE std::string StringPrintf(const c8* fmt, Args &&...args);

	template <typename... Args>
	SSSP_FORCE_INLINE std::string StringPrintf(const c8* fmt, Args &&...args)
	{
		std::string ret;
		Internal::StringPrintfRecursive(&ret, fmt, std::forward<Args>(args)...);
		return ret;
	}

	template <typename... Args>
	void Printf(const c8* fmt, Args &&...args)
	{
		std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
		fputs(s.c_str(), stdout);
	}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif  // __GNUG__

	// https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
	SSSP_FORCE_INLINE std::string Red(const std::string& s)
	{
		const c8* red = "\033[1m\033[31m";  // bold red
		const c8* reset = "\033[0m";
		return std::string(red) + s + std::string(reset);
	}

	SSSP_FORCE_INLINE std::string Yellow(const std::string& s)
	{
		// https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
		const c8* yellow = "\033[1m\033[38;5;100m";
		const c8* reset = "\033[0m";
		return std::string(yellow) + s + std::string(reset);
	}

	SSSP_FORCE_INLINE std::string Green(const std::string& s)
	{
		// https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
		const c8* green = "\033[1m\033[38;5;22m";
		const c8* reset = "\033[0m";
		return std::string(green) + s + std::string(reset);
	}

}


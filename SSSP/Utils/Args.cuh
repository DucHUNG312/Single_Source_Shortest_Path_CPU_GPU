#pragma once

#include <Core/Core.cuh>
#include <Utils/Print.cuh>
#include <Utils/Options.cuh>

#ifdef SSSP_PLATFORM_WINDOWS
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <windows.h>
#include <shellapi.h>
#endif  // SSSP_PLATFORM_WINDOWS

namespace SSSP
{
	bool Atoi(const std::string& str, i32* ptr);
	bool Atoi(const std::string& str, i64* ptr);
	bool Atof(const std::string& str, f32* ptr);
	bool Atof(const std::string& str, f64* ptr);

	std::vector<std::string> SplitStringsFromWhitespace(const std::string& str);
	std::vector<std::string> SplitString(const std::string& str, c8 ch);
	std::vector<i32> SplitStringToInts(const std::string& str, c8 ch);
	std::vector<i64> SplitStringToInt64s(const std::string& str, c8 ch);
	std::vector<f32> SplitStringToFloats(const std::string& str, c8 ch);
	std::vector<f64> SplitStringToDoubles(const std::string& str, c8 ch);

#ifdef SSSP_PLATFORM_WINDOWS
	std::wstring WStringFromU16String(std::u16string str);
	std::wstring WStringFromUTF8(std::string str);
	std::u16string U16StringFromWString(std::wstring str);
	std::string UTF8FromWString(std::wstring str);
#endif  // SSSP_PLATFORM_WINDOWS

	// https://stackoverflow.com/a/52703954
	std::string UTF8FromUTF16(std::u16string str);
	std::u16string UTF16FromUTF8(std::string str);

	SSSP_FORCE_INLINE std::string NormalizeArg(const std::string& str)
	{
		std::string ret;
		for (unsigned char c : str)
		{
			if (c != '_' && c != '-')
				ret += std::tolower(c);
		}
		return ret;
	}

	SSSP_FORCE_INLINE bool MatchPrefix(const std::string& str, const std::string& prefix)
	{
		if (prefix.size() > str.size())
			return false;
		for (size_t i = 0; i < prefix.size(); ++i)
			if (prefix[i] != str[i])
				return false;
		return true;
	}

	SSSP_FORCE_INLINE bool InitArg(const std::string& str, i32* ptr) {
		if (str.empty() || (!std::isdigit(str[0]) && str[0] != '-'))
			return false;
		try
		{
			*ptr = Atoi(str, ptr);
		}
		catch (const std::invalid_argument&)
		{
			return false;
		}
		catch (const std::out_of_range&)
		{
			return false;
		}
		return true;
	}

	SSSP_FORCE_INLINE bool InitArg(const std::string& str, f32* ptr) {
		if (str.empty())
			return false;
		try
		{
			*ptr = Atof(str, ptr);
		}
		catch (const std::invalid_argument&)
		{
			return false;
		}
		catch (const std::out_of_range&)
		{
			return false;
		}
		return true;
	}

	SSSP_FORCE_INLINE bool InitArg(const std::string& str, f64* ptr) {
		if (str.empty())
			return false;
		try
		{
			*ptr = Atof(str, ptr);
		}
		catch (const std::invalid_argument&)
		{
			return false;
		}
		catch (const std::out_of_range&)
		{
			return false;
		}
		return true;
	}

	SSSP_FORCE_INLINE bool InitArg(const std::string& str, c8** ptr)
	{
		if (str.empty())
			return false;
		*ptr = new c8[str.size() + 1];
		std::strcpy(*ptr, str.data());
		return true;
	}

	SSSP_FORCE_INLINE bool InitArg(const std::string& str, std::string* ptr) {
		if (str.empty())
			return false;
		*ptr = str;
		return true;
	}

	SSSP_FORCE_INLINE bool InitArg(const std::string& str, bool* ptr) {
		if (NormalizeArg(str) == "false")
		{
			*ptr = false;
			return true;
		}
		else if (NormalizeArg(str) == "true")
		{
			*ptr = true;
			return true;
		}
		return false;
	}

	template <typename T>
	SSSP_FORCE_INLINE bool Enable(T ptr)
	{
		return false;
	}

	SSSP_FORCE_INLINE bool Enable(bool* ptr)
	{
		*ptr = true;
		return true;
	}

	SSSP_FORCE_INLINE std::vector<std::string> GetCommandLinesArgs(c8** argv);

	// T basically needs to be a pointer type or a Span.
	template <typename Iter, typename T>
	SSSP_FORCE_INLINE bool ParseArg(Iter* iter, Iter end, const std::string& name, T out, std::function<void(std::string)> onError)
	{
		std::string arg = **iter;

		// Strip either one or two leading dashes.
		if (arg[1] == '-')
			arg = arg.substr(2);
		else
			arg = arg.substr(1);

		if (MatchPrefix(NormalizeArg(arg), NormalizeArg(name + '='))) 
		{
			// --arg=value
			std::string value = arg.substr(name.size() + 1);
			if (!InitArg(value, out)) 
			{
				onError(StringPrintf("invalid value \"%s\" for --%s argument", value, name));
				return false;
			}
			return true;
		}
		else if (NormalizeArg(arg) == NormalizeArg(name)) 
		{
			// --arg <value>, except for bool arguments, which are set to true
			// without expecting another argument.
			if (Enable(out))
				return true;

			++(*iter);
			if (*iter == end) 
			{
				onError(StringPrintf("missing value after --%s argument", arg));
				return false;
			}
			if (!InitArg(**iter, out)) 
			{
				onError(StringPrintf("invalid value \"%s\" for --%s argument", **iter, name));
				return false;
			}
			return true;
		}
		else
			return false;
	}

	class CLIParser
	{
	public:
		static void Usage(const std::string& msg = {});
		static Options Parse(i32 argc, c8** argv);
	};
}
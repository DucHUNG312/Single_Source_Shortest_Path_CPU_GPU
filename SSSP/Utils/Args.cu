#pragma once

#include <Utils/Args.cuh>

namespace SSSP
{
	bool Atoi(const std::string& str, i32* ptr)
	{
		try
		{
			*ptr = std::stoi(std::string(str.begin(), str.end()));
		}
		catch (...)
		{
			return false;
		}
		return true;
	}

	bool Atoi(const std::string& str, i64* ptr)
	{
		try
		{
			*ptr = std::stoll(std::string(str.begin(), str.end()));
		}
		catch (...)
		{
			return false;
		}
		return true;
	}

	bool Atof(const std::string& str, f32* ptr)
	{
		try
		{
			*ptr = std::stof(std::string(str.begin(), str.end()));
		}
		catch (...)
		{
			return false;
		}
		return true;
	}

	bool Atof(const std::string& str, f64* ptr)
	{
		try
		{
			*ptr = std::stod(std::string(str.begin(), str.end()));
		}
		catch (...)
		{
			return false;
		}
		return true;
	}

	std::vector<std::string> SplitStringsFromWhitespace(const std::string& str) {
		std::vector<std::string> ret;

		i32 start = 0;
		while (start < str.size()) 
		{
			// skip leading whitespace
			while (start < str.size() && std::isspace(str[start])) 
			{
				++start;
			}

			// find the end of the current word
			auto end = start;
			while (end < str.size() && !std::isspace(str[end])) 
			{
				++end;
			}

			ret.push_back(str.substr(start, end - start));
			start = end;
		}

		return ret;
	}

	std::vector<std::string> SplitString(const std::string& str, c8 ch) 
	{
		std::vector<std::string> strings;

		if (str.empty()) 
		{
			return strings;
		}

		i32 start = 0;
		while (start < str.size()) 
		{
			auto end = str.find(ch, start);
			if (end == std::string::npos) 
			{
				strings.push_back(str.substr(start));
				break;
			}
			else 
			{
				strings.push_back(str.substr(start, end - start));
				start = end + 1;
			}
		}

		return strings;
	}

	std::vector<i32> SplitStringToInts(const std::string& str, c8 ch)
	{
		std::vector<std::string> strs = SplitString(str, ch);
		std::vector<i32> i32s(strs.size());

		for (size_t i = 0; i < strs.size(); ++i)
			if (!Atoi(strs[i], &i32s[i]))
				return {};
		return i32s;
	}

	std::vector<i64> SplitStringToInt64s(const std::string& str, c8 ch)
	{
		std::vector<std::string> strs = SplitString(str, ch);
		std::vector<i64> i32s(strs.size());

		for (size_t i = 0; i < strs.size(); ++i)
			if (!Atoi(strs[i], &i32s[i]))
				return {};
		return i32s;
	}

	std::vector<f32> SplitStringToFloats(const std::string& str, c8 ch)
	{
		std::vector<std::string> strs = SplitString(str, ch);
		std::vector<f32> f32s(strs.size());

		for (size_t i = 0; i < strs.size(); ++i)
			if (!Atof(strs[i], &f32s[i]))
				return {};
		return f32s;
	}

	std::vector<f64> SplitStringToDoubles(const std::string& str, c8 ch)
	{
		std::vector<std::string> strs = SplitString(str, ch);
		std::vector<f64> f64s(strs.size());

		for (size_t i = 0; i < strs.size(); ++i)
			if (!Atof(strs[i], &f64s[i]))
				return {};
		return f64s;
	}

#ifdef SSSP_PLATFORM_WINDOWS
	std::wstring WStringFromU16String(std::u16string str)
	{
		std::wstring ws;
		ws.reserve(str.size());
		for (cw16 c : str)
			ws.push_back(c);
		return ws;
	}

	std::wstring WStringFromUTF8(std::string str)
	{
		return WStringFromU16String(UTF16FromUTF8(str));
	}

	std::u16string U16StringFromWString(std::wstring str)
	{
		std::u16string su16;
		su16.reserve(str.size());
		for (c16 c : str)
			su16.push_back(c);
		return su16;
	}

	std::string UTF8FromWString(std::wstring str)
	{
		return UTF8FromUTF16(U16StringFromWString(str));
	}

#endif  // SSSP_PLATFORM_WINDOWS

	// https://stackoverflow.com/a/52703954
	std::string UTF8FromUTF16(std::u16string str)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<c16, 0x10ffff, std::codecvt_mode::little_endian>, c16> cnv;
		std::string utf8 = cnv.to_bytes(str);
		SSSP_CHECK_GE(cnv.converted(), str.size());
		return utf8;
	}

	std::u16string UTF16FromUTF8(std::string str)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<c16, 0x10ffff, std::codecvt_mode::little_endian>, c16> cnv;
		std::u16string utf16 = cnv.from_bytes(str);
		SSSP_CHECK_GE(cnv.converted(), str.size());
		return utf16;
	}

	std::vector<std::string> GetCommandLinesArgs(c8** argv)
	{
		std::vector<std::string> argStrs;
#ifdef SSSP_PLATFORM_WINDOWS
		i32 argc;
		LPWSTR* argvw = CommandLineToArgvW(GetCommandLineW(), &argc);
		SSSP_CHECK(argv != nullptr);
		for (i32 i = 1; i < argc; ++i)
			argStrs.push_back(UTF8FromWString(argvw[i]));
#else
		++argv;
		while (argv)
		{
			argStrs.push_back(*argv);
			++argv;
		}
#endif  // SSSP_PLATFORM_WINDOWS
		return argStrs;
	}

	void CLIParser::Usage(const std::string& msg)
	{
		if (!msg.empty())
			fprintf(stderr, "SSSP: %s\n\n", msg.c_str());

		fprintf(stderr,
			R"(usage: SSSP [<options>] <filename.txt...>

Options:
  --cpu							Use the CPU path.
  --gpu							Use the GPU path.
  --hybrid						Use CPU GPU hybrid path (Experiment).
  --nthreads <num>              Use specified number of threads for OpenMP.
  --dataFile <filename>         Use specified dataset file path.
)");
		exit(msg.empty() ? 0 : 1);
	}

	Options CLIParser::Parse(i32 argc, c8** argv)
	{
		std::vector<std::string> args = GetCommandLinesArgs(argv);
		Options options;

		// Process command-line arguments
		for (auto iter = args.begin(); iter != args.end(); ++iter)
		{
			auto onError = [](const std::string& err) {
				Usage(err);
				exit(1);
				};

			if (
				ParseArg(&iter, args.end(), "cpu", &options.cpu, onError) ||
				ParseArg(&iter, args.end(), "gpu", &options.gpu, onError) ||
				ParseArg(&iter, args.end(), "hybrid", &options.hybrid, onError) ||
				ParseArg(&iter, args.end(), "nthreads", &options.numThreadOpenMP, onError) ||
				ParseArg(&iter, args.end(), "dataFile", &options.dataFile, onError)
				) {
				// success
			}
			else if (*iter == "--help" || *iter == "-help" || *iter == "-h")
			{
				Usage();
				exit(0);
			}
			else
			{
				Usage(StringPrintf("argument \"%s\" unknown", *iter));
				exit(1);
			}
		}

		// Print banner
#ifdef SSSP_DEBUG_BUILD
		SSSP_LOG_INFO_NL("*** RUNNING DEBUG BUILD ***\n");
#endif
		SSSP_LOG_INFO_NL("Copyright (c)2023 Le Vu Duc Hung.");
		SSSP_LOG_INFO_NL("The source code is covered by the MIT License.");
		SSSP_LOG_INFO_NL("See the file LICENSE.txt for the conditions of the license.");
		fflush(stdout);

		// Check validity of provided arguments
		if (options.dataFile.empty())
		{
			SSSP_LOG_CRITICAL_NL("Data file path is required!");
			exit(1);
		}
		if (!(options.cpu || options.gpu || options.hybrid))
		{
			SSSP_LOG_CRITICAL_NL("Must use at least 2 options in 3 (cpu, gpu, hybrid)");
			exit(1);
		}
		if (options.numThreadOpenMP && !options.cpu)
		{
			SSSP_LOG_CRITICAL_NL("The --nthreads option is only supported in cpu mode");
			exit(1);
		}

		return options;
	}
}

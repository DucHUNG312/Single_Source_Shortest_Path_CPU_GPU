#include <Utils/Print.cuh>
#include <External/double-conversion/double-conversion.h>

namespace SSSP
{
	namespace Internal
	{
		std::string FloatToString(f32 v)
		{
			double_conversion::DoubleToStringConverter doubleConverter(
				double_conversion::DoubleToStringConverter::NO_FLAGS, "Inf", "NaN", 'e',
				-6 /* decimal_in_shortest_low */, 9 /* decimal_in_shortest_high */,
				5 /* max_leading_padding_zeroes_in_precision_mode */,
				5 /*  max_trailing_padding_zeroes_in_precision_mode */);
			c8 buf[64];
			double_conversion::StringBuilder result(buf, SSSP_ARRAYSIZE(buf));
			doubleConverter.ToShortestSingle(v, &result);
			i32 length = result.position();
			return std::string(buf, length);
		}

		std::string DoubleToString(f64 v)
		{
			double_conversion::DoubleToStringConverter doubleConverter(
				double_conversion::DoubleToStringConverter::NO_FLAGS, "Inf", "NaN", 'e',
				-6 /* decimal_in_shortest_low */, 9 /* decimal_in_shortest_high */,
				5 /* max_leading_padding_zeroes_in_precision_mode */,
				5 /*  max_trailing_padding_zeroes_in_precision_mode */);
			c8 buf[64];
			double_conversion::StringBuilder result(buf, SSSP_ARRAYSIZE(buf));
			doubleConverter.ToShortest(v, &result);
			i32 length = result.position();
			return std::string(buf, length);
		}

		void StringPrintfRecursive(std::string* s, const c8* fmt)
		{
			const c8* c = fmt;
			// No args left; make sure there aren't any extra formatting
			// specifiers.
			while (*c)
			{
				if (*c == '%')
				{
					if (c[1] != '%')
						SSSP_LOG_CRITICAL_NL("Not enough optional values passed to Printf.");
					++c;
				}
				*s += *c++;
			}
		}

		// 1. Copy from fmt to *s, up to the next formatting directive.
		// 2. Advance fmt past the next formatting directive and return the
		//    formatting directive as a string.
		std::string CopyToFormatString(const c8** fmt_ptr, std::string* s)
		{
			const c8*& fmt = *fmt_ptr;
			while (*fmt)
			{
				if (*fmt != '%')
				{
					*s += *fmt;
					++fmt;
				}
				else if (fmt[1] == '%')
				{
					// "%%"; let it pass through
					*s += '%';
					*s += '%';
					fmt += 2;
				}
				else
					// fmt is at the start of a formatting directive.
					break;
			}

			std::string nextFmt;
			while (*fmt)
			{
				c8 c = *fmt;
				nextFmt += c;
				++fmt;
				// Is it a conversion specifier?
				if (c == 'd' || c == 'i' || c == 'o' || c == 'u' || c == 'x' || c == 'e' ||
					c == 'E' || c == 'f' || c == 'F' || c == 'g' || c == 'G' || c == 'a' ||
					c == 'A' || c == 'c' || c == 'C' || c == 's' || c == 'S' || c == 'p')
					break;
			}

			return nextFmt;
		}
	}
}

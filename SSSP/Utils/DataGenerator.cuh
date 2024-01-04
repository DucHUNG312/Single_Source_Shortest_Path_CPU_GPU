#pragma once

#include <Core/Core.cuh>
#include <Utils/Random.cuh>

/// This class only used for generate data

namespace SSSP
{
	class DataGenerator
	{
	public:
		static void Generator(i32 minVertex, i32 maxVertex, const std::string& filepath, i32 maxConnection = 15)
		{
			RandGenerator rand;
			File::CreateFileIfNotExist(filepath);
			std::ofstream outputStream;
			try
			{
				outputStream.open(filepath);
			}
			catch (const std::exception& e)
			{
				std::cerr << "Error opening session: " << e.what() << '\n';
			}


			for (size_t i = minVertex; i <= maxVertex; i++)
			{
				i32 iter = rand.GenerateInt(0, maxConnection);

				for (size_t j = 0; j < iter; j++)
				{
					i32 connectSource = rand.GenerateInt(minVertex, maxVertex);
					outputStream << i << "\t" << connectSource << '\n';
				}
			}

			try
			{
				outputStream.close();
			}
			catch (const std::exception& e)
			{
				std::cerr << "Error closing session: " << e.what() << '\n';
			}
		}
	};

}
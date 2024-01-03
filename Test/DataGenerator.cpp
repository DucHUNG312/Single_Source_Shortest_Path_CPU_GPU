#include <iostream>
#include<inttypes.h>
#include<string>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Windows.h>


using i32 = int32_t;
using f32 = float;

class RandGenerator
{
public:
    RandGenerator()
    {
    }

    f32 GenerateReal()
    {
        return distribution(engine);
    }


    i32 GenerateInt(i32 min, i32 max)
    {
        std::uniform_int_distribution<i32> intDistribution(min, max);
        return intDistribution(engine);
    }
private:
    std::random_device rd;
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<f32> distribution{ 0.f, 1.f };
};

class File
{
public:
	static void CreateFileIfNotExist(const std::string& filePath)
	{
		std::ifstream fileStream(filePath);
		if (!fileStream.is_open()) 
		{
			std::ofstream newFile(filePath);
			if (newFile.is_open()) 
			{
				std::cout << "File created: " << filePath << std::endl;
				newFile.close();
			}
			else 
			{
				std::cerr << "Error: Could not create the file." << std::endl;
			}
		}
		else 
		{
			std::cout << "File already exists: " << filePath << std::endl;
			fileStream.close();
		}
	}
};


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

i32 main()
{
	DataGenerator::Generator(0, 1000000, "E://CPUGPU//DataSets//data1.txt");
}

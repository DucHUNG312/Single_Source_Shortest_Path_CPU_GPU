#pragma once

#ifdef SSSP_PLATFORM_WINDOWS
#include <Windows.h>
#endif // SSSP_PLATFORM_WINDOWS

namespace SSSP
{
	class File
	{
	public:
		static std::string GetCurrentExecutablePath()
		{
#ifdef SSSP_PLATFORM_WINDOWS
			c8 buffer[MAX_PATH];
			GetModuleFileNameA(nullptr, buffer, MAX_PATH);
			std::string path(buffer);
			return path.substr(0, path.find_last_of("\\/"));
#else
			SSSP_UNIMPLEMENTED;
			return std::string();
#endif // SSSP_PLATFORM_WINDOWS
		}

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

		static void CreateFileEvenAlreadyExist(const std::string& filePath)
		{
			std::ifstream fileStream(filePath);
			if (!fileStream.is_open())
			{
				std::ofstream newFile(filePath, std::ios::trunc); // Use std::ios::trunc to clear content if the file exists
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
				fileStream.close();
				std::ofstream existingFile(filePath, std::ios::trunc); // Open existing file and clear content
				if (existingFile.is_open())
				{
#ifdef SSSP_DEBUG
					std::cout << "Cleard existing file content and write: " << filePath << std::endl;
#endif //SSSP_DEBUG
					existingFile.close();
				}
				else
				{
					std::cerr << "Error: Could not clear content of the existing file." << std::endl;
				}
			}
		}


		static std::string ReadDataFromFile(const std::string& filePath) 
		{
			std::ifstream fileStream(filePath);
			std::string fileContent;

			if (fileStream.is_open()) 
			{
				std::string line;

				while (std::getline(fileStream, line)) 
				{
					fileContent += line + "\n";
				}
				fileStream.close();
			}
			else 
			{
				std::cerr << "Error: Unable to open file " << filePath << std::endl;
			}

			return fileContent;
		}

		static std::vector<std::string> ReadFromFileVector(const std::string& filePath)
		{
			std::ifstream fileStream(filePath);
			std::vector<std::string> fileContent;

			if (fileStream.is_open())
			{
				std::string line;
				while (std::getline(fileStream, line))
				{
					fileContent.push_back(line);
				}
				fileStream.close();
			}
			else
			{
				std::cerr << "Error: Unable to open file " << filePath << std::endl;
			}

			return fileContent;
		}
	};
}
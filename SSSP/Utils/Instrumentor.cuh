/// This class is use for output json profiler file to use in chrome://tracing

#pragma once

#include <Utils/File.cuh>

namespace SSSP
{
    using FloatingPointMicroseconds = std::chrono::duration<f64, std::micro>;

    struct InstrumentationSession
    {
        std::string Name;
    };

    struct ProfileResult
    {
        std::string Name;
        FloatingPointMicroseconds Start;
        std::chrono::microseconds ElapsedTime;
        std::thread::id ThreadID;
    };

    class Instrumentor
    {
    public:
        Instrumentor()
            : currentSession(nullptr), profileCount(0)
        {
        }

        ~Instrumentor()
        {
            EndSession();
        }

        void BeginSession(const std::string& name, const std::string& filepath)
        {
            File::CreateFileEvenAlreadyExist(filepath);
            // Only one session can be run at a time
            if (currentSession) { EndSession(); }
            currentSession = new InstrumentationSession{ name };

            try { outputStream.open(filepath); }
            catch (const std::exception& e) { std::cerr << "Error opening session: " << e.what() << '\n'; }

            WriteHeader();
        }

        void EndSession()
        {
            // Only one session can be run at a time
            if (!currentSession) { return; }

            WriteFooter();

            try { outputStream.close(); }
            catch (const std::exception& e) { std::cerr << "Error closing session: " << e.what() << '\n'; }

            delete currentSession;
            currentSession = nullptr;
            profileCount = 0;
        }

        void WriteProfile(const ProfileResult& result)
        {
            std::lock_guard<std::mutex> lock(m_Lock);

            if (profileCount++ > 0)
                outputStream << ",";

            std::string name = result.Name;
            std::replace(name.begin(), name.end(), '"', '\'');

            outputStream << std::setprecision(3) << std::fixed;
            outputStream << "{";
            outputStream << "\"cat\":\"function\",";
            outputStream << "\"dur\":" << (result.ElapsedTime.count()) << ',';
            outputStream << "\"name\":\"" << name << "\",";
            outputStream << "\"ph\":\"X\",";
            outputStream << "\"pid\":0,";
            outputStream << "\"tid\":" << result.ThreadID << ",";
            outputStream << "\"ts\":" << result.Start.count();
            outputStream << "}";
        }

        void WriteHeader()
        {
            outputStream << "{\"otherData\": {},\"traceEvents\":[";
        }

        void WriteFooter()
        {
            outputStream << "]}";
        }

        static Instrumentor& Get()
        {
            static Instrumentor instance;
            return instance;
        }
    private:
        InstrumentationSession* currentSession;
        std::ofstream outputStream;
        int profileCount;
        std::mutex m_Lock;
    };

    class InstrumentationTimer
    {
    public:
        InstrumentationTimer(const c8* name)
            : name(name), stopped(false)
        {
            Start();
        }

        ~InstrumentationTimer()
        {
            if (!stopped)
                Stop();
        }

        void PrintTimer(const std::string& funcName)
        {
            Update();
            SSSP_LOG_DEBUG_NL("[{}] took {} ms.", funcName, elapsedTime.count() / 1000);
        }

        const c8* GetName() { return name; }
        std::chrono::time_point<std::chrono::steady_clock> GetStartTimepoint() { return startTimepoint; }
        bool IsStopped() { return stopped; }
        std::chrono::steady_clock::time_point GetEndTimepoint() { return endTimepoint; }
        FloatingPointMicroseconds GetHighResStart() { return highResStart; }
        std::chrono::duration<long long, std::micro> GetElapsedTime() { return elapsedTime; }
    private:
        void Update()
        {
            endTimepoint = std::chrono::steady_clock::now();
            highResStart = FloatingPointMicroseconds{ startTimepoint.time_since_epoch() };
            elapsedTime = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch() - std::chrono::time_point_cast<std::chrono::microseconds>(startTimepoint).time_since_epoch();
        }

        void Start()
        {
            startTimepoint = std::chrono::steady_clock::now();
        }

        void Stop()
        {
            Update();
            Instrumentor::Get().WriteProfile({ name, highResStart, elapsedTime, std::this_thread::get_id() });
            stopped = true;
        }
    private:
        const c8* name;
        std::chrono::time_point<std::chrono::steady_clock> startTimepoint;
        bool stopped;
        std::chrono::steady_clock::time_point endTimepoint;
        FloatingPointMicroseconds highResStart;
        std::chrono::duration<long long, std::micro> elapsedTime;
    };
}
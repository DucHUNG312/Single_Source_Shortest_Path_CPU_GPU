#include <Utils/Memory.cuh>

namespace SSSP
{
    CPUMemMonitor* CPUMemMonitor::s_MemoryMonitor = nullptr;

    CPUMemMonitor::CPUMemMonitor()
    {
        s_MemoryMonitor = this;

        PdhOpenQuery(NULL, NULL, &cpuQuery);

        PdhAddEnglishCounterW(cpuQuery, L"\\Processor(_Total)\\% Processor Time", NULL, &cpuTotal);
        PdhCollectQueryData(cpuQuery);

        SYSTEM_INFO sysInfo;
        FILETIME ftime, fsys, fuser;

        GetSystemInfo(&sysInfo);
        numProcessors = sysInfo.dwNumberOfProcessors;

        GetSystemTimeAsFileTime(&ftime);
        memcpy(&lastCPU, &ftime, sizeof(FILETIME));

        currentprocess = GetCurrentProcess();
        GetProcessTimes(currentprocess, &ftime, &ftime, &fsys, &fuser);
        memcpy(&lastSysCPU, &fsys, sizeof(FILETIME));
        memcpy(&lastUserCPU, &fuser, sizeof(FILETIME));
    }

    CPUMemMonitor::~CPUMemMonitor()
    {
        if (cpuQuery == NULL)
        {
            PdhCloseQuery(cpuQuery);
        }
    }

    CPUUse CPUMemMonitor::GetCPUUsage()
    {
        PDH_FMT_COUNTERVALUE counterVal;
        PdhCollectQueryData(cpuQuery);
        PdhGetFormattedCounterValue(cpuTotal, PDH_FMT_DOUBLE, NULL, &counterVal);
        CPUUse c;
        c.TotalUse = counterVal.doubleValue;

        FILETIME ftime, fsys, fuser;
        ULARGE_INTEGER now, sys, user;
        f64 percent = 0.0;
        GetSystemTimeAsFileTime(&ftime);
        memcpy(&now, &ftime, sizeof(FILETIME));
        GetProcessTimes(currentprocess, &ftime, &ftime, &fsys, &fuser);
        memcpy(&sys, &fsys, sizeof(FILETIME));
        memcpy(&user, &fuser, sizeof(FILETIME));
        percent = static_cast<f64>(sys.QuadPart - lastSysCPU.QuadPart) + (user.QuadPart - lastUserCPU.QuadPart);
        percent /= static_cast<f64>(now.QuadPart - lastCPU.QuadPart);
        percent /= static_cast<f64>(numProcessors);
        lastCPU = now;
        lastUserCPU = user;
        lastSysCPU = sys;
        c.ProcessUse = percent * 100.0;
        return c;
    }

    MemoryUse CPUMemMonitor::GetMemoryUsage()
    {
        PROCESS_MEMORY_COUNTERS_EX pmc;
        HANDLE curProcess = GetCurrentProcess();
        GetProcessMemoryInfo(curProcess, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        MemoryUse m;
        m.PhysicalTotalUsed = memInfo.ullTotalPhys - memInfo.ullAvailPhys;
        m.PhysicalTotalAvailable = memInfo.ullTotalPhys;
        m.PhysicalProcessUsed = pmc.WorkingSetSize;

        m.VirtualTotalAvailable = memInfo.ullTotalPageFile;
        m.VirtualTotalUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;
        m.VirtualProcessUsed = pmc.PrivateUsage;
        return m;
    }
}
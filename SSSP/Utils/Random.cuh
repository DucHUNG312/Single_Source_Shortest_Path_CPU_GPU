#include <Core/Core.cuh>

namespace SSSP
{
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
        std::mt19937 engine{ rd() };
        std::uniform_real_distribution<f32> distribution{ 0.f, 1.f };
    };
}
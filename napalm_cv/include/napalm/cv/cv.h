#pragma once
#include <napalm/napalm.h>

namespace napalm
{
    namespace cv
    {
        void convolv(napalm::Context * ctx, napalm::Buffer * input, napalm::Buffer * output,
            int width, int height, int channel_count,
            napalm::Buffer * kernel_data, int k_width, int k_height, int stream_idx = 0);
    }
}
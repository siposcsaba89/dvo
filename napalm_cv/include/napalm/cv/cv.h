#pragma once
#include <napalm/napalm.h>

namespace napalm
{
    namespace cv
    {
        void convolv(napalm::Context * ctx, const napalm::Buffer * input, napalm::Buffer * output,
            int width, int height, int channel_count,
            const napalm::Buffer * kernel_data, int k_width, int k_height, int stream_idx = 0);
        void convolv(napalm::Context * ctx, const napalm::Img * input, napalm::Img * output,
            const napalm::Buffer * kernel_data, int k_width, int k_height, int stream_idx = 0);
    }
}
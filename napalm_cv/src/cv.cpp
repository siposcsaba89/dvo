#include "../include/napalm/cv/cv.h"
#include <napalm/gen/programs/convolv.h>
namespace napalm
{
    namespace cv{
        void convolv(napalm::Context * ctx, const napalm::Buffer * input, napalm::Buffer * output, 
            int width, int height, int channel_count, 
            const napalm::Buffer * kernel_data, int k_width, int k_height, int stream_idx)
        {
            napalm::gen::convolv convolv(*napalm::ProgramStore::create(ctx));

            const int loc_size_x = 16;
            const int loc_size_y = 16;
            if (k_width == 3 && k_height == 3)
            {
                convolv("convolv_buff_3x3")(stream_idx,
                    napalm::ImgRegion((width + loc_size_x - 1) / loc_size_x, (height + loc_size_y - 1) / loc_size_y),
                    napalm::ImgRegion(loc_size_x, loc_size_y),
                    input, output, width, height, channel_count, kernel_data);
            }
            else if (k_width == 5 && k_height == 5)
            {
                convolv("convolv_buff_5x5")(stream_idx,
                    napalm::ImgRegion((width + loc_size_x - 1) / loc_size_x, (height + loc_size_y - 1) / loc_size_y),
                    napalm::ImgRegion(loc_size_x, loc_size_y),
                    input, output, width, height, channel_count, kernel_data);
            }
            else
            {
                convolv("convolv_buff_nxn")(stream_idx,
                    napalm::ImgRegion((width + loc_size_x - 1) / loc_size_x, (height + loc_size_y - 1) / loc_size_y),
                    napalm::ImgRegion(loc_size_x, loc_size_y),
                    input, output, width, height, channel_count, kernel_data, k_width, k_height);
            }
        }
        void convolv(napalm::Context * ctx, const napalm::Img * input, napalm::Img * output, const napalm::Buffer * kernel_data, int k_width, int k_height, int stream_idx)
        {
            napalm::gen::convolv convolv(*napalm::ProgramStore::create(ctx));
            int32_t w = output->img_size.x;
            int32_t h = output->img_size.y;
            const int loc_size_x = 16;
            const int loc_size_y = 16;
            if (k_width == 3 && k_height == 3)
            {
                convolv("convolv_img_3x3")(stream_idx,
                    napalm::ImgRegion((w + loc_size_x - 1) / loc_size_x, 
                    (h + loc_size_y - 1) / loc_size_y),
                    napalm::ImgRegion(loc_size_x, loc_size_y),
                    input, output, w, h, kernel_data);
            }
            else if (k_width == 5 && k_height == 5)
            {
                convolv("convolv_img_5x5")(stream_idx,
                    napalm::ImgRegion((w + loc_size_x - 1) / loc_size_x, 
                    (h + loc_size_y - 1) / loc_size_y),
                    napalm::ImgRegion(loc_size_x, loc_size_y),
                    input, output, w, h, kernel_data);
            }
            else
            {
                convolv("convolv_img_nxn")(stream_idx,
                    napalm::ImgRegion((w + loc_size_x - 1) / loc_size_x, 
                    (h + loc_size_y - 1) / loc_size_y),
                    napalm::ImgRegion(loc_size_x, loc_size_y),
                    input, output, w, h, kernel_data, k_width, k_height);
            }
        }
    }
}

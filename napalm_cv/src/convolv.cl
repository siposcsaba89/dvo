

inline void convolv_buff(global const unsigned char* input, global unsigned char * output, 
    int width, int height, int chc, global const float * kernel_data, const int k_width, const int k_height)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= width || j >= height)
    {
        return;
    }
    else if (j - k_height / 2 < 0 || i - k_width / 2 < 0)
    {
        for (int k = 0; k < chc; ++k)
        {
            output[chc * (width * j + i) + k] = 0;
        }
    }
    else
    {
        for (int k = 0; k < chc; ++k)
        {
            float res = 0.0f;
#pragma unroll
            for (int y = 0; y < k_height; ++y)
            {
#pragma unroll
                for (int x = 0; x < k_width; ++x)
                {
                    res += kernel_data[y * k_width + x] * input[chc * (width * (j - k_height / 2 + y) + i - k_width / 2 + x) + k];
                }

            }
            output[chc * (width * j + i) + k] = (unsigned char) res;
        }
    }

}

kernel void convolv_buff_3x3(global const unsigned char* input, global unsigned char * output,
    int width, int height, int chc, global const float * kernel_data)
{
    convolv_buff(input, output, width, height, chc, kernel_data, 3, 3);
}

kernel void convolv_buff_5x5(global const unsigned char* input, global unsigned char * output,
    int width, int height, int chc, global const float * kernel_data)
{
    convolv_buff(input, output, width, height, chc, kernel_data, 5, 5);
}

kernel void convolv_buff_nxn(global const unsigned char* input, global unsigned char * output,
    const int width, const int height, const int chc, 
    global const float * kernel_data, const int k_width, const int k_height)
{
    convolv_buff(input, output, width, height, chc, kernel_data, k_width, k_height);
}



inline void convolv_img(read_only image2d_t input, write_only image2d_t output,
    const int img_w, const int img_h, 
    global const float * kernel_data, const int k_width, const int k_height)
{
    const sampler_t img_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    float i = get_global_id(0);
    float j = get_global_id(1);

    if (i >= img_w || j >= img_h)
        return;

    float4 res = 0.0f;
#pragma unroll
    for (int y = 0; y < k_height; ++y)
    {
#pragma unroll
        for (int x = 0; x < k_width; ++x)
        {
            res += read_imagef(input, img_sampler, (float2)(i - k_width / 2 + x + 0.5f, (j - k_height / 2 + y) + 0.5f));
        }

    }
    write_imagef(output, (int2)(i, j), res);
}

kernel void convolv_img_3x3(read_only image2d_t input, write_only image2d_t output,
    const int img_w, const int img_h,
    global const float * kernel_data)
{
    convolv_img(input, output, img_w, img_h, kernel_data, 3, 3);
}

kernel void convolv_img_5x5(read_only image2d_t input, write_only image2d_t output,
    const int img_w, const int img_h,
    global const float * kernel_data)
{
    convolv_img(input, output, img_w, img_h, kernel_data, 5, 5);
}

kernel void convolv_img_nxn(read_only image2d_t input, write_only image2d_t output,
    const int img_w, const int img_h,
    global const float * kernel_data, const int k_width, const int k_height)
{
    convolv_img(input, output, img_w, img_h, kernel_data, k_width, k_height);
}


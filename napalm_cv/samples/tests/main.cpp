#include <iostream>
#include <vector>
#include <napalm/cv/cv.h>

using namespace std;

template <typename T>
bool checkRes(const std::vector<T> & out, int width, int height, int chc, int val)
{
    bool ret = true;
    for (int j = 1; j < height - 1; ++j)
    {
        for (int i = 1; i < width - 1; ++i)
        {
            for (int c = 0; c < chc; ++c)
            {
                ret = ret && out[(j * width + i) *chc + c] == val;
                if (out[(j * width + i) *chc + c] != val)
                {
                    printf("Error, %d != %d \n", int(out[(j * width + i) *chc + c]), val);
                }
            }
        }

    }
    return ret;
}


int main()
{
    int w = 640;
    int h = 360;
    int chc = 1;

    vector<uint8_t> img_data(w * h * chc, 1);
    vector<float> k_data(9, 1.0f);

    napalm::Context * ctx = napalm::createContext("OpenCL", 2, 0, 3);
    napalm::Buffer * kenel = ctx->createBuffer(3 * 3 * sizeof(float),
        napalm::MEM_FLAG_READ_WRITE | napalm::MEM_FLAG_COPY_HOST_PTR, k_data.data());
    //Buffer version
    {
        napalm::Buffer * in = ctx->createBuffer(w * h * chc, napalm::MEM_FLAG_READ_WRITE | napalm::MEM_FLAG_COPY_HOST_PTR,
            img_data.data());
        napalm::Buffer * out = ctx->createBuffer(w * h * chc);

        napalm::cv::convolv(ctx, in, out, w, h, chc, kenel, 3, 3);


        vector<uint8_t> img_data_out(w * h * chc);
        out->read(img_data_out.data());

        if (checkRes(img_data_out, w, h, chc, 9))
        {
            printf("Buffer test success! \n");
        }
        else
        {
            printf("Buffer test unsuccess! \n");

        }
        delete in;
        delete out; 
    }
    ctx->finish(0);
    //image version
    {
        napalm::Img * in = ctx->createImg(napalm::ImgFormat(), napalm::ImgRegion(w,h),
            napalm::MEM_FLAG_READ_WRITE | napalm::MEM_FLAG_COPY_HOST_PTR,
            img_data.data());
        ctx->finish(0);

        napalm::Img * out = ctx->createImg(napalm::ImgFormat(napalm::IMG_CHANNEL_FORMAT_INTENSITY,
            napalm::DATA_TYPE_UNORM_INT8), napalm::ImgRegion(w,h));
        ctx->finish(0);

        napalm::cv::convolv(ctx, in, out, kenel, 3, 3);
        ctx->finish(0);

        vector<uint8_t> img_data_out(w * h * chc);
        out->read(img_data_out.data());

        if (checkRes(img_data_out, w, h, chc, 9))
        {
            printf("Img test success! \n");
        }
        else
        {
            printf("Img test unsuccess! \n");

        }

        delete in;
        delete out;
    }
    delete kenel;
    napalm::destroyContext( ctx);
    return 0;
}
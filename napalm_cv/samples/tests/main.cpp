#include <iostream>
#include <vector>
#include <napalm/cv/cv.h>

using namespace std;

int main()
{
    int w = 640;
    int h = 360;
    int chc = 1;

    vector<uint8_t> img_data(w * h * chc, 1);

    napalm::Context * ctx = napalm::createContext("OpenCL", 0, 0, 3);

    napalm::Buffer * in = ctx->createBuffer(w * h * chc, napalm::MEM_FLAG_READ_WRITE, img_data.data());
    in->write(img_data.data());
    napalm::Buffer * out = ctx->createBuffer(w * h * chc);
    napalm::Buffer * kenel = ctx->createBuffer(3 * 3 * sizeof(float));

    vector<float> k_data(9, 1.0f);

    kenel->write(k_data.data());

    napalm::cv::convolv(ctx, in, out, w, h, chc, kenel, 3, 3);


    vector<uint8_t> img_data_out(w * h * chc);
    out->read(img_data_out.data());


    delete in;
    delete out;
    delete kenel;
    return 0;
}
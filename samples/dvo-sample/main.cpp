#include <iostream>
#include <opencv2/opencv.hpp>
#include <napalm/napalm.h>
#include <dvo/dvo.h>

napalm::Context * createContext(const std::string & backend, int platform_idx, int device_idx);

int main(int argc, const char *  argv[])
{
    std::string cmd_args =
        "{h ? help | | Application help message}"
        "{@left | | Left input source name}"
        "{@right | | Right input source name}"
        "{i intrinsics | intrinsics.yml | Camera calibration intrinsics file name}"
        "{e extrinsics | extrinsics.yml | Camera calibration extrinsics file name}"
        "{s start_idx | 0 | Start the video from this frame}"
        "{b backend | OpenCL | Could be OpenCL or CUDA}"
        "{p platform_idx | 0 | OpenCL platform idx, -1 - choose from command line, 0..num_platform -1  - real platforms as listed by the opencl runtime}"
        "{d device_idx | 0 | OpenCL device idx, 0..num_devices-1}"
        "{c config | config.yml | dvo config file}";

    cv::CommandLineParser parser(argc, argv, cmd_args);

    if (parser.has("h"))
    {
        parser.printMessage();
        exit(EXIT_SUCCESS);
    }

    std::string left_source_name = parser.get<std::string>(0);
    std::string right_source_name = parser.get<std::string>(1);
    if (left_source_name == "" || right_source_name == "")
    {
        std::cout << "Left and right input source must not be empty!" << std::endl;
        parser.printMessage();
        exit(EXIT_FAILURE);
    }

    std::string intrinsics_name = parser.get<std::string>("i");
    std::string extrinsics_name = parser.get<std::string>("e");
    int start_frame = parser.get<int>("s");
    std::string napalm_backend = parser.get<std::string>("b");
    if (napalm_backend != "OpenCL" && napalm_backend != "CUDA")
    {
        std::cout << "Napalm backend must be CUDA or OpenCL!" << std::endl;
        exit(EXIT_FAILURE);
    }
    int platform_idx = parser.get<int>("p");
    int device_idx = parser.get<int>("d");
    std::string config_name = parser.get<std::string>("c");

    cv::VideoCapture l_cap(left_source_name);
    cv::VideoCapture r_cap(right_source_name);

    if (!l_cap.isOpened())
    {
        std::cout << "Cannot Open left image source: " << left_source_name << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!r_cap.isOpened())
    {
        std::cout << "Cannot Open right image source: " << right_source_name << std::endl;
        exit(EXIT_FAILURE);
    }


    cv::Mat left_in, right_in;
    bool do_exit = false;
    bool is_paused = false;

    napalm::Context * n_ctx = createContext(napalm_backend, platform_idx, device_idx);

    while (!do_exit)
    {
        if (!is_paused)
        {
            do_exit = !l_cap.read(left_in) || !r_cap.read(right_in);
        }
        cv::imshow("left", left_in);
        cv::imshow("right", right_in);
        int key = cv::waitKey(1);

        if (key == 27)
            do_exit = true;
        else if (key == ' ')
            is_paused = !is_paused;
    }

    napalm::destroyContext(n_ctx);
    return 0;
}


napalm::Context * createContext(const std::string & backend, int platform_idx, int device_idx)
{
    napalm::PlatformAndDeviceInfo * info = napalm::getPlatformAndDeviceInfo(backend.c_str());
    if (info->num_platforms == 0)
    {
        std::cout << "Cannot find any platform, exiting!" << std::endl;
        delete info;
        exit(EXIT_FAILURE);
    }
    if (platform_idx == -1)
    {
        while (platform_idx < 0 || platform_idx >= info->num_platforms)
        {
            std::cout << "Available platforms: " << std::endl;
            for (int32_t i = 0; i < info->num_platforms; ++i)
            {
                std::cout << "\t" << i << ": " << info->platforms[i] << std::endl;
            }

            std::cout << "Select a platform: ";
            std::string str;
            std::cin >> str;
            try
            {
             platform_idx = std::stoi(str);
            } catch(std::exception e)
            {
                platform_idx = -1;
            }
        }
        device_idx = -1;
        while (device_idx < 0 || device_idx >= info->num_devices[platform_idx])
        {
            std::cout << "Available devices: " << std::endl;
            for (int32_t i = 0; i < info->num_devices[platform_idx]; ++i)
            {
                std::cout << "\t" << i << ": " << info->device_names[platform_idx][i] << std::endl;
            }
            std::cout << "Select a device: ";
            std::string str;
            std::cin >> str;
            try
            {
                device_idx = std::stoi(str);
            }
            catch (std::exception e)
            {
                device_idx = -1;
            }
        }


    }

    if (platform_idx >= 0 && platform_idx < info->num_platforms)
    {
        std::cout << "Selected platform: " << info->platforms[platform_idx] << std::endl;

        if (device_idx >= 0 && device_idx < info->num_devices[platform_idx])
            std::cout << "Selected device: " << info->device_names[platform_idx][device_idx] << std::endl;
        else
        {
            std::cout << "Invalid device selected: " << device_idx << std::endl;
            std::cout << "device must be in: 0.." << info->num_devices[platform_idx] << std::endl;
            delete info;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cout << "Invalid platform selected: " << platform_idx << std::endl;
        std::cout << "Platform idx must be in: 0.." << info->num_platforms << std::endl;
        delete info;
        exit(EXIT_FAILURE);
    }
    delete info;
    return napalm::createContext(backend.c_str(), platform_idx, device_idx, 3);
}

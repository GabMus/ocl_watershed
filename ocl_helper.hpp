#pragma once

#include <CL/cl.hpp>
#include <string>

std::string read_kernel(std::string kernel_path) {
    // Read the kernel file and return it as string;
    std::ifstream sourceFile(kernel_path);
    if (!sourceFile) {
        std::cerr << TERM_RED << "Error: Could not find kernel source for path " << kernel_path << std::endl << TERM_RESET;
        exit(1);
    }
    std::string sourceCode((std::istreambuf_iterator<char>(sourceFile)), (std::istreambuf_iterator<char>()));
    sourceFile.close();
    return sourceCode;
}

cl::Device ocl_get_default_device() {
    // get all platforms
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    int howmany_platforms = all_platforms.size();
    if (howmany_platforms == 0) {
        std::cerr << "No platforms found. Make sure OpenCL is installed correctly.\n";
        exit(1);
    }

    // Platform slection
    std::cout << "Platforms found:\n";
    for (int i=0; i<howmany_platforms; i++) {
        std::cout << "#" << i << ": " << all_platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using default platform " <<
                 TERM_BOLD <<
                 default_platform.getInfo<CL_PLATFORM_NAME>() <<
                 TERM_RESET <<
                 std::endl;

    // Device selection
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    int howmany_devices = all_devices.size();
    if (howmany_devices == 0){
        std::cerr << "No devices found. Make sure OpenCL is installed correctly.\n";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    std::cout << "Using default device " <<
                TERM_BOLD <<
                default_device.getInfo<CL_DEVICE_NAME>() <<
                TERM_RESET <<
                std::endl;
    return default_device;
}

double profile_kernel(
        cl::CommandQueue &queue,
        cl::Kernel &kernel,
        cl::NDRange offset,
        cl::NDRange global,
        cl::NDRange local,
        std::string message="") {
    
    cl::Event event;
    queue.enqueueNDRangeKernel(
                kernel,
                offset,
                global,
                local,
                NULL, //This is `const VECTOR_CLASS<Event>* events` and defaults to NULL anyway
                &event);
    event.wait();
    queue.finish();
    
    cl_ulong time_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    //std::cout << "\tDEBUG: "<< time_start << std::endl << "\t       " << time_end << std::endl;

    double nanoseconds = time_end-time_start;
    double milliseconds = nanoseconds/1000000.0;

    std::cout << TERM_CYAN <<
            message <<
            std::setprecision(5) <<
            milliseconds << "ms" <<
            TERM_RESET << std::endl;

    return milliseconds;
}

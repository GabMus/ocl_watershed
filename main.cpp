#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
// These are for the random generation
#include <cstdlib>
#include <ctime>
// For log2f and floor/ceiling
#include <cmath>
#include <CL/cl.hpp>

#define TERM_BOLD "\033[1m"
#define TERM_GREEN "\x1b[32m"
#define TERM_RED "\x1b[31m"
#define TERM_CYAN "\033[36m"
#define TERM_RESET "\x1b[0m"

#define DEBUG 1

#include "cl_errorcheck.hpp"
#include "include/cxxopts.hpp"
#include "imagelib.hpp"
#include "io_helper.hpp"
#include "ocl_helper.hpp"

int main(int argc, const char** argv) {

    std::string pwd = get_dir(argv[0]);

    cxxopts::Options options("ocl_watershed", "OpenCL implementation of the watershed transform");
    options.add_options()
        ("p,profiling", "Enable profiling")
        ("i,input", "Input PPM image file path",
            cxxopts::value<std::string>())
        ("o,output", "Output file path",
            cxxopts::value<std::string>()->default_value(pwd + "/out.ppm"));

    auto result = options.parse(argc, argv);

    std::string bmp_path="";
    std::string out_path="";

    if (result.count("i") == 1) { 
        bmp_path = result["i"].as<std::string>();
    }
    else {
        std::cout << options.help() << std::endl;
        exit(1);
    }

    out_path = result["o"].as<std::string>();
    bool enable_profiling = result.count("p");

    if (enable_profiling) std::cout << TERM_CYAN <<
        "Running with profiling enabled" << TERM_RESET << std::endl;

    cl_int err;

    BMPVEC bmp;

    int bmp_width;
    int bmp_height;

    read_ppm(bmp_path, bmp, bmp_width, bmp_height);

    std::cout << TERM_GREEN <<
                 "Loaded picture: " <<
                 bmp_path << std::endl <<
                 "    Size: " <<
                 bmp_width << "x" << bmp_height <<
                 TERM_RESET << std::endl;


    BMPVEC bmp_RGBA_data;
    bgr2bgra(bmp, bmp_RGBA_data);

    cl::Device default_device = ocl_get_default_device();
    cl::Context context({default_device});
    cl::Program::Sources sources;

    cl::CommandQueue queue;
    if (enable_profiling) queue = cl::CommandQueue(context, default_device, CL_QUEUE_PROFILING_ENABLE);
    else queue = cl::CommandQueue(context, default_device);

    cl::Image2D cl_input_image =  cl::Image2D(
                context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                bmp_width, bmp_height,
                0,
                (void*)(&bmp_RGBA_data[0]),
                &err);
    cl_check(err, "Creating input image");

    cl::Image2D cl_luma_image = cl::Image2D(
                context,
                CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating luma image");

    cl::Image2D cl_gradient_image = cl::Image2D(
                context,
                CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating gradient image");

    cl::Image2D cl_output_image = cl::Image2D(
                context,
                CL_MEM_WRITE_ONLY,
                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating output image");

    cl::Buffer cl_t0_lattice(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);
    cl::Buffer cl_t1_lattice(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);

    cl::Buffer cl_t0_labels(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);
    cl::Buffer cl_t1_labels(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);
    
    uint32_t* host_init_are_diff = new uint32_t();
    host_init_are_diff[0] = 0u;
    cl::Buffer cl_are_diff(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_uint), host_init_are_diff, &err);

    cl_check(err, "Creating are_diff value buffer");

    std::string ocl_source = read_kernel(pwd + "/ocl_source.cl");
    sources.push_back({ocl_source.c_str(), ocl_source.length()});
    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << TERM_RED <<
                     "Error Building: " <<
                     program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) <<
                     TERM_RESET << std::endl;
        exit(1);
    }

    cl::Kernel kernel_make_luma_image = cl::Kernel(program, "make_luma_image");
    cl::Kernel kernel_make_gradient = cl::Kernel(program, "make_gradient");
    cl::Kernel kernel_init_t0 = cl::Kernel(program, "init_t0");
    cl::Kernel kernel_automaton = cl::Kernel(program, "automaton");
    cl::Kernel kernel_color_watershed = cl::Kernel(program, "color_watershed");

    kernel_make_luma_image.setArg(0, cl_input_image);
    kernel_make_luma_image.setArg(1, cl_luma_image);

    queue.enqueueNDRangeKernel(
            kernel_make_luma_image,
            cl::NullRange,
            cl::NDRange(bmp_width, bmp_height),
            cl::NullRange);

    queue.finish();

    kernel_make_gradient.setArg(0, cl_luma_image);
    kernel_make_gradient.setArg(1, cl_gradient_image);
    
    queue.enqueueNDRangeKernel(
            kernel_make_gradient,
            cl::NullRange,
            cl::NDRange(bmp_width, bmp_height),
            cl::NullRange);

    queue.finish();

    kernel_init_t0.setArg(0, cl_t0_lattice);
    kernel_init_t0.setArg(1, cl_t0_labels);
    kernel_init_t0.setArg(2, bmp_width);
    kernel_init_t0.setArg(3, cl_gradient_image);

    queue.enqueueNDRangeKernel(
                kernel_init_t0,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);

    queue.finish();

    kernel_automaton.setArg(0, cl_luma_image);
    kernel_automaton.setArg(1, bmp_width);
    kernel_automaton.setArg(2, bmp_height);
    kernel_automaton.setArg(7, cl_are_diff);

    for (int i=0; i<=std::max(bmp_width, bmp_height); i++) {
        kernel_automaton.setArg(3, cl_t0_lattice);
        kernel_automaton.setArg(4, cl_t0_labels);
        kernel_automaton.setArg(5, cl_t1_lattice);
        kernel_automaton.setArg(6, cl_t1_labels);

        host_init_are_diff[0] = 0;
        queue.enqueueWriteBuffer(cl_are_diff, CL_TRUE, 0, sizeof(cl_uint), host_init_are_diff);
        queue.finish();


        // Insert profiling here
        if (enable_profiling) {
            profile_kernel(
                        queue,
                        kernel_automaton,
                        cl::NullRange,
                        cl::NDRange(bmp_width, bmp_height),
                        cl::NullRange,
                        "Step " + std::to_string(i) + ": ");
        }
        else {
            queue.enqueueNDRangeKernel(
                        kernel_automaton,
                        cl::NullRange,
                        cl::NDRange(bmp_width, bmp_height),
                        cl::NullRange);
            queue.finish();
        }

        queue.enqueueReadBuffer(cl_are_diff, CL_TRUE, 0, sizeof(uint32_t), host_init_are_diff);
        queue.finish();

        if (!host_init_are_diff[0])  {
            std::cout << TERM_CYAN <<
                "Baling out early from automaton loop at step #" << i <<
                std::endl << TERM_RESET;
            break;
        }

        std::swap(cl_t0_labels, cl_t1_labels);
        std::swap(cl_t0_lattice, cl_t1_lattice);
    }

    queue.finish();

    kernel_color_watershed.setArg(0, cl_input_image);
    kernel_color_watershed.setArg(1, bmp_width);
    kernel_color_watershed.setArg(2, bmp_height);
    kernel_color_watershed.setArg(3, cl_t1_labels);
    kernel_color_watershed.setArg(4, cl_output_image);

    queue.enqueueNDRangeKernel(
                    kernel_color_watershed,
                    cl::NullRange,
                    cl::NDRange(bmp_width, bmp_height),
                    cl::NullRange);

    queue.finish();

    uint8_t* host_outimage = new uint8_t[bmp_width*bmp_height*4];
    uint8_t* rgb_outimage = new uint8_t[bmp_width*bmp_height*3];

    cl::size_t<3> ri_origin;
    ri_origin[0] = 0;
    ri_origin[1] = 0;
    ri_origin[2] = 0;
    cl::size_t<3> ri_region;
    ri_region[0] = bmp_width;
    ri_region[1] = bmp_height;
    ri_region[2] = 1;
    err = queue.enqueueReadImage(
                        cl_output_image,
                        CL_TRUE,
                        ri_origin,
                        ri_region,
                        0,
                        0,
                        host_outimage);
    cl_check(err, "Reading image from device");

    rgba2rgb(
        host_outimage,
        bmp_width*bmp_height,
        rgb_outimage
    );

    write_ppm(rgb_outimage,
        3*bmp_width*bmp_height,
        bmp_width,
        bmp_height,
        out_path);

    return 0;
}

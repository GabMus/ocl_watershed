#include <iostream>
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
#include "imagelib.hpp"
#include "io_helper.hpp"
#include "ocl_helper.hpp"

#define BMP_PATH "/home/gabmus/test.ppm"

int main(int argc, char** argv) {

    std::string bmp_path="";

    if (argc < 2) {
        std::cout << "Image path not provided, falling back to " << BMP_PATH << std::endl;
        bmp_path = BMP_PATH;
    }
    else {
        bmp_path = argv[1];
    }

    cl_int err;

    std::string pwd = get_dir(argv[0]);

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

    //BMPVEC bmp_BGR_data;
    BMPVEC bmp_RGBA_data;
    //get_bitmap_data(bmp, bmp_BGR_data);
    bgr2bgra(bmp,
            bmp_RGBA_data);

    cl::Device default_device = ocl_get_default_device();
    cl::Context context({default_device});
    cl::Program::Sources sources;
    cl::CommandQueue queue(context, default_device);

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

    cl::Image2D cl_output_image = cl::Image2D(
                context,
                CL_MEM_WRITE_ONLY,
                cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating output image");

    cl::Buffer cl_minima_value(context, CL_MEM_READ_WRITE, sizeof(uint32_t));

    cl::Buffer cl_t0_lattice(context, CL_MEM_READ_WRITE, sizeof(uint32_t)*bmp_width*bmp_height);
    cl::Buffer cl_t1_lattice(context, CL_MEM_READ_WRITE, sizeof(uint32_t)*bmp_width*bmp_height);

    cl::Buffer cl_t0_labels(context, CL_MEM_READ_WRITE, sizeof(uint32_t)*bmp_width*bmp_height);
    cl::Buffer cl_t1_labels(context, CL_MEM_READ_WRITE, sizeof(uint32_t)*bmp_width*bmp_height);

    std::string ocl_source = read_kernel(pwd + "/ocl_source.cl");
    sources.push_back({ocl_source.c_str(), ocl_source.length()});
    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << TERM_RED <<
                     "Error Building: " <<
                     program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) <<
                     TERM_RESET <<
                     std::endl;
        exit(1);
    }

    cl::Kernel kernel_init_globals = cl::Kernel(program, "init_globals");
    cl::Kernel kernel_find_minima = cl::Kernel(program, "find_minima");
    cl::Kernel kernel_init_t0 = cl::Kernel(program, "init_t0");
    cl::Kernel kernel_automaton = cl::Kernel(program, "automaton");

    kernel_init_globals.setArg(0, cl_minima_value);

    kernel_find_minima.setArg(0, cl_input_image);
    kernel_find_minima.setArg(1, cl_luma_image);
    kernel_find_minima.setArg(2, cl_minima_value);

    kernel_init_t0.setArg(0, cl_t0_lattice);
    kernel_init_t0.setArg(1, cl_t0_labels);
    kernel_init_t0.setArg(2, bmp_width);
    kernel_init_t0.setArg(3, cl_luma_image);
    kernel_init_t0.setArg(4, cl_minima_value);

    queue.enqueueNDRangeKernel(
                kernel_init_globals,
                cl::NullRange,
                cl::NDRange(1),
                cl::NullRange);

    queue.enqueueNDRangeKernel(
                kernel_find_minima,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);

    queue.enqueueNDRangeKernel(
                kernel_init_t0,
                cl::NullRange,
                cl::NDRange(bmp_width*bmp_height),
                cl::NullRange);


    for ()

    kernel_automaton.setArg(0, cl_luma_image);
    kernel_automaton.setArg(1, );
    kernel_automaton.setArg(2, );
    kernel_automaton.setArg(3, );

    queue.enqueueNDRangeKernel(
                kernel_automaton,
                cl::NullRange,
                cl::NDRange(bmp_width*bmp_height),
                cl::NullRange);





    queue.finish();

    uint32_t* host_outvec = new uint32_t[bmp_width*bmp_height];

    queue.enqueueReadBuffer(cl_t0_lattice, CL_TRUE, 0, sizeof(uint32_t)*bmp_width*bmp_height, host_outvec);

    uint8_t* host_outimage = new uint8_t[bmp_width*bmp_height];

    cl::size_t<3> ri_origin;
    ri_origin[0] = 0;
    ri_origin[1] = 0;
    ri_origin[2] = 0;
    cl::size_t<3> ri_region;
    ri_region[0] = bmp_width;
    ri_region[1] = bmp_height;
    ri_region[2] = 1;
    err = queue.enqueueReadImage(
                        cl_luma_image,
                        CL_TRUE,
                        ri_origin,
                        ri_region,
                        0,
                        0,
                        host_outimage);
    cl_check(err, "Reading image from device");

    uint8_t* rgb_pixelvec =  new uint8_t[bmp_width*bmp_height*3];

    r2rgb((unsigned char*)host_outimage,
        bmp_width*bmp_height,
        (unsigned char*)rgb_pixelvec);

    write_ppm(rgb_pixelvec,
        3*bmp_width*bmp_height,
        bmp_width,
        bmp_height,
        "/home/gabmus/ocl_out_fuck.ppm");

    return 0;
}

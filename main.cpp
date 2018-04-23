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

#define BMP_PATH "/home/gabmus/Development/ocl_watershed_misc/redflowers100.ppm"

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


    BMPVEC bmp_RGBA_data;
    bgr2bgra(bmp, bmp_RGBA_data);

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
                cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating output image");

    uint32_t* host_minima_value = new uint32_t();
    *host_minima_value = 256u;

    cl::Buffer cl_minima_value(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(uint32_t), host_minima_value, &err);

    cl_check(err, "Creating minima value buffer");

    cl::Buffer cl_t0_lattice(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);
    cl::Buffer cl_t1_lattice(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);

    cl::Buffer cl_t0_labels(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);
    cl::Buffer cl_t1_labels(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*bmp_width*bmp_height);

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

    //cl::Kernel kernel_init_globals = cl::Kernel(program, "init_globals");
    cl::Kernel kernel_make_luma_image = cl::Kernel(program, "make_luma_image");
    cl::Kernel kernel_make_gradient = cl::Kernel(program, "make_gradient");
    cl::Kernel kernel_find_minima = cl::Kernel(program, "find_minima");
    cl::Kernel kernel_init_t0 = cl::Kernel(program, "init_t0");
    cl::Kernel kernel_automaton = cl::Kernel(program, "automaton");

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

    kernel_find_minima.setArg(0, cl_gradient_image);
    kernel_find_minima.setArg(1, cl_minima_value);

    queue.enqueueNDRangeKernel( // OLD: NOT ANYMORE //~~~also builds cl_luma_image~~~
                kernel_find_minima,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);

#if 1
    //DEBUG
    queue.finish();
    queue.enqueueReadBuffer(cl_minima_value, CL_TRUE, 0, sizeof(uint32_t), host_minima_value);
    queue.finish();
    std::cout << TERM_CYAN <<
        "DEBUG: minima_value after find_minima: " <<
        *host_minima_value <<
        std::endl <<
        TERM_RESET;
#endif

    kernel_init_t0.setArg(0, cl_t0_lattice);
    kernel_init_t0.setArg(1, cl_t0_labels);
    kernel_init_t0.setArg(2, bmp_width);
    kernel_init_t0.setArg(3, cl_gradient_image);
    kernel_init_t0.setArg(4, cl_minima_value);

    queue.enqueueNDRangeKernel(
                kernel_init_t0,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);

#if 1

    queue.finish();
    uint32_t* out_lattice = new uint32_t[bmp_width*bmp_height];
    queue.enqueueReadBuffer(cl_t0_lattice, CL_TRUE, 0, sizeof(uint32_t)*bmp_width*bmp_height, out_lattice);

    uint8_t* r8_pixelvec_lattice = new uint8_t[bmp_width*bmp_height];
    r32_2_r8(
            out_lattice,
            bmp_width*bmp_height,
            r8_pixelvec_lattice
    );
    uint8_t* rgb_pixelvec_lattice =  new uint8_t[bmp_width*bmp_height*3];

    r2rgb((unsigned char*)r8_pixelvec_lattice,
        bmp_width*bmp_height,
        (unsigned char*)rgb_pixelvec_lattice);

    /*write_ppm(rgb_pixelvec_lattice,
        3*bmp_width*bmp_height,
        bmp_width,
        bmp_height,
        "/home/gabmus/watershed_misc/ocl_out_lattice.ppm");*/

#endif

    for (int i=0; i<=std::max(bmp_width, bmp_height)*50; i++) {
        //std::cout << TERM_GREEN << "i: " << i << " - " << "0 --> 1\n" << TERM_RESET;
        kernel_automaton.setArg(0, cl_luma_image);
        kernel_automaton.setArg(1, bmp_width);
        kernel_automaton.setArg(2, bmp_height);
        kernel_automaton.setArg(3, bmp_width*bmp_height);
        kernel_automaton.setArg(4, cl_t0_lattice);
        kernel_automaton.setArg(5, cl_t0_labels);
        kernel_automaton.setArg(6, cl_t1_lattice);
        kernel_automaton.setArg(7, cl_t1_labels);

        queue.enqueueNDRangeKernel(
                    kernel_automaton,
                    cl::NullRange,
                    cl::NDRange(bmp_width*bmp_height),
                    cl::NullRange);

        queue.finish();
        std::swap(cl_t0_labels, cl_t1_labels);
        std::swap(cl_t0_lattice, cl_t1_lattice);
    }

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
        "/home/gabmus/Development/ocl_watershed_misc/ocl_out.ppm");

    uint32_t* out_labels = new uint32_t[bmp_width*bmp_height];
    queue.enqueueReadBuffer(cl_t1_labels, CL_TRUE, 0, sizeof(uint32_t)*bmp_width*bmp_height, out_labels);

    uint8_t* r_outimage = new uint8_t[bmp_width*bmp_height];
    uint8_t* rgb_outimage = new uint8_t[bmp_width*bmp_height*3];

    color_watershed(out_labels, (uint8_t*)host_outimage, bmp_width, bmp_height, r_outimage);

    r2rgb(r_outimage, bmp_width*bmp_height, rgb_outimage);

    write_ppm(rgb_outimage,
        3*bmp_width*bmp_height,
        bmp_width,
        bmp_height,
        "/home/gabmus/Development/ocl_watershed_misc/ocl_out_watershed.ppm");

    return 0;
}

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

#define HOST_VECINIT 0
#define DEBUG 1

#include "cl_errorcheck.hpp"
#include "imagelib.hpp"
#include "io_helper.hpp"
#include "ocl_helper.hpp"

#define BMP_PATH "/home/gabmus/Desktop/pics/panorama.ppm"

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
                cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
                bmp_width, bmp_height,
                0,
                (void*)(&bmp_RGBA_data[0]),
                &err);
    cl_check(err, "Creating input image");

    cl::Image2D cl_blurred_image =  cl::Image2D(
                context,
                CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating blurred image");

    cl::Image2D cl_sobelx_image = cl::Image2D(
                context,
                CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_R, CL_UNORM_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating sobelx image");

    cl::Image2D cl_sobely_image = cl::Image2D(
                context,
                CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_R, CL_UNORM_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating sobely image");

    cl::Image2D cl_output_image = cl::Image2D(
                context,
                CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_R, CL_UNORM_INT8),
                bmp_width, bmp_height,
                0,
                NULL,
                &err);
    cl_check(err, "Creating output image");

    std::string kernel_code_sobel = read_kernel(pwd + "/sobel_kernel.cl");
    std::string kernel_code_gaussian = read_kernel(pwd + "/gaussian_kernel.cl");
    std::string kernel_code_zone = read_kernel(pwd + "/zone_kernel.cl");
    sources.push_back({kernel_code_sobel.c_str(), kernel_code_sobel.length()});
    sources.push_back({kernel_code_gaussian.c_str(), kernel_code_gaussian.length()});
    sources.push_back({kernel_code_zone.c_str(), kernel_code_zone.length()});
    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << TERM_RED <<
                     "Error Building: " <<
                     program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) <<
                     TERM_RESET <<
                     std::endl;
        exit(1);
    }

    cl::Kernel kernel_sobelx = cl::Kernel(program, "sobelx");
    cl::Kernel kernel_sobely = cl::Kernel(program, "sobely");
    cl::Kernel kernel_sum_sobel = cl::Kernel(program, "sum_sobel");
    cl::Kernel kernel_gaussian = cl::Kernel(program, "gaussian");
    cl::Kernel kernel_zone = cl::Kernel(program, "zone");

    kernel_gaussian.setArg(0, cl_input_image);
    kernel_gaussian.setArg(1, cl_blurred_image);

    kernel_sobelx.setArg(0, cl_blurred_image);
    kernel_sobelx.setArg(1, cl_sobelx_image);

    kernel_sobely.setArg(0, cl_blurred_image);
    kernel_sobely.setArg(1, cl_sobely_image);

    kernel_sum_sobel.setArg(0, cl_sobelx_image);
    kernel_sum_sobel.setArg(1, cl_sobely_image);
    kernel_sum_sobel.setArg(2, cl_output_image);

    queue.enqueueNDRangeKernel(
                kernel_gaussian,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);
    queue.finish();

    queue.enqueueNDRangeKernel(
                kernel_sobelx,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);
    queue.enqueueNDRangeKernel(
                kernel_sobely,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);
    queue.finish();

    queue.enqueueNDRangeKernel(
                kernel_sum_sobel,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);
    queue.finish();

    unsigned char* host_outvec = new unsigned char[bmp_width*bmp_height];
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
                    host_outvec);
    cl_check(err, "Reading image from device");

    unsigned char* rgb_pixelvec =  new unsigned char[bmp_width*bmp_height*3];
    r2rgb(
                host_outvec,
                bmp_width*bmp_height,
                rgb_pixelvec);

    write_ppm(rgb_pixelvec,
              3*bmp_width*bmp_height,
              bmp_width,
              bmp_height,
              "/home/gabmus/ocl_out.ppm");

#if DO_ZONE
    kernel_zone.setArg(0, cl_input_image);
    kernel_zone.setArg(1, cl_output_image);
    queue.enqueueNDRangeKernel(
                kernel_zone,
                cl::NullRange,
                cl::NDRange(bmp_width, bmp_height),
                cl::NullRange);
    queue.finish();

    err = queue.enqueueReadImage(
                    cl_output_image,
                    CL_TRUE,
                    ri_origin,
                    ri_region,
                    0,
                    0,
                    host_outvec);
    cl_check(err, "Reading image from device");

    r2rgb(
                host_outvec,
                bmp_width*bmp_height,
                rgb_pixelvec);

    write_ppm(rgb_pixelvec,
              3*bmp_width*bmp_height,
              bmp_width,
              bmp_height,
              "/home/gabmus/ocl_out_zone.ppm");

#endif

    return 0;
}

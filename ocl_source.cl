#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


#define R_LUMA_MULT 0.2126f
#define G_LUMA_MULT 0.7152f
#define B_LUMA_MULT 0.0722f
#define MAX_INT 4294967295

void kernel init_globals(global uint* minima_value) {
    minima_value[0]=255u;
}

void kernel find_minima(
    read_only image2d_t in_pic,
    write_only image2d_t luma_pic,
    global uint* minima_value) {

    uint4 pixel = read_imageui(in_pic, (int2){get_global_id(0), get_global_id(1)});
    uint luma = floor((R_LUMA_MULT * pixel.x) + (G_LUMA_MULT * pixel.y) + (B_LUMA_MULT * pixel.z));
    atomic_min(minima_value, luma); // does the following in one operation (should grant mutual exclusion): luma < &minima_value ? {&minima_value=luma} : pass;
    write_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}, luma);
}

void kernel init_t0(
    global uint* t0_lattice,
    global uint* t0_labels,
    int width,
    read_only image2d_t luma_pic,
    global uint* minima_value) {

    uint pixval = read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x;
    uint pos = get_global_id(0)+(get_global_id(1)*width);
    t0_lattice[pos] = pixval <= *minima_value ? (uint)0 : (uint)MAX_INT;
    t0_labels[pos] = pixval <= *minima_value ? (uint)pos : (uint)0;
}

void kernel automaton(
    global read_only image2d_t in_pic,
    global const uint32_t* t0_lattice,
    global uint32_t* t1_lattice) { // a same image cannot be rw

    u_t

    read_imagef(in_pic, gauss_sampler, (int2){get_global_id(0)+i-1, get_global_id(1)+j-1}) * ck_gauss_5x[i+(j*5)];
    float4 n_pixel = (float4){0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int i=0; i<5; i++) {
        #pragma unroll
        for (int j=0; j<5; j++) {
            n_pixel += read_imagef(in_pic, gauss_sampler, (int2){get_global_id(0)+i-1, get_global_id(1)+j-1}) * ck_gauss_5x[i+(j*5)];
        }
    }

    write_imagef(out_pic, (int2){get_global_id(0), get_global_id(1)},  n_pixel);
}

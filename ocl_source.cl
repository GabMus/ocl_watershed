#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define R_LUMA_MULT 0.2126f
#define G_LUMA_MULT 0.7152f
#define B_LUMA_MULT 0.0722f
#define MAX_INT 4294967295

//void kernel init_globals(global uint* minima_value) {
//    *minima_value=255u;
//}

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

//void kernel automaton(
//    global read_only image2d_t luma_pic, // this contains the values for f(p)
//    int width,
//    int size,
//    global const uint32_t* t0_lattice,
//    global const uint32_t* t0_labels,
//    global uint32_t* t1_lattice,
//    global uint32_t* t1_labels) { // a same image cannot be rw

//    uint pos = get_global_id(0)+(get_global_id(1)*width);
//    // x: north, y: east, z: south, t: west
//    /*
//    WARNING: there is a problem with this implementation:
//             MAX_INT + (any number) re-cycles back from 0!
//             It goes beyond the uint32_t limit.
//             There MUST be a better way to say "this neighbor doesn't exist"
//             NEEDS FURTHER INVESTIGATION
//    */
//    uint4 u_t_neighborhood = (uint4){
//        get_global_id(1) != 0 ? t0_lattice[pos-width] : MAX_INT, // exists if it's not the first row
//        get_global_id(0) != (width-1) ? t0_lattice[pos+1] : MAX_INT, // exists if it's not the last column
//        get_global_id(1) != (height-1) ? asd : MAX_INT, // exists if it's not the last row
//        get_global_id(0) != 0 ? t0_lattice[pos-1] : MAX_INT, //exists if it's not the first column
//    } + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x;

//    read_imagef(in_pic, gauss_sampler, (int2){get_global_id(0)+i-1, get_global_id(1)+j-1}) * ck_gauss_5x[i+(j*5)];
//    float4 n_pixel = (float4){0.0f, 0.0f, 0.0f, 0.0f};

//    #pragma unroll
//    for (int i=0; i<5; i++) {
//        #pragma unroll
//        for (int j=0; j<5; j++) {
//            n_pixel += read_imagef(in_pic, gauss_sampler, (int2){get_global_id(0)+i-1, get_global_id(1)+j-1}) * ck_gauss_5x[i+(j*5)];
//        }
//    }

//    write_imagef(out_pic, (int2){get_global_id(0), get_global_id(1)},  n_pixel);
//}

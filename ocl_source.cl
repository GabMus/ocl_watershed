#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define R_LUMA_MULT 0.2126f
#define G_LUMA_MULT 0.7152f
#define B_LUMA_MULT 0.0722f
#define MAX_INT 9999

//void kernel init_globals(global uint* minima_value) {
//    *minima_value=255u;
//}

/**/
constant int ck_gradientx[9] = { // gradient horizontal
	-1,	-1,	-1,
	0,	0,	0,
	1,	1,	1
};
/**/

/*
constant int ck_gradientx[9] = { // Edge
	0,	-1,	0,
	-1,	4,	-1,
	0,	-1,	0
};
*/

void kernel make_luma_image(
    read_only image2d_t in_pic,
    write_only image2d_t luma_pic) {

    uint4 pixel = read_imageui(in_pic, (int2){get_global_id(0), get_global_id(1)}).x;
    uint luma = floor((R_LUMA_MULT * pixel.x) + (G_LUMA_MULT * pixel.y) + (B_LUMA_MULT * pixel.z));
    write_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}, luma);
}

void kernel make_gradient(
    read_only image2d_t in_pic, // this should be luma_pic
    write_only image2d_t gradient_pic) {

    int n_pixel = 0;

    #pragma unroll
    for (int i=0; i<3; i++) {
        #pragma unroll
        for (int j=0; j<3; j++) {
            uint luma = read_imageui(in_pic, sampler, (int2){get_global_id(0)+i-1, get_global_id(1)+j-1}).x;
            n_pixel += (int)luma * ck_gradientx[i+(j*3)];
        }
    }

    write_imageui(gradient_pic, (int2){get_global_id(0), get_global_id(1)}, n_pixel);

}

void kernel find_minima(
    read_only image2d_t in_pic, // this should be gradient_pic
    global uint* minima_value) {

    uint luma = read_imageui(in_pic, (int2){get_global_id(0), get_global_id(1)}).x;
    atomic_min(minima_value, luma); // does the following in one operation (should grant mutual exclusion): luma < &minima_value ? {&minima_value=luma} : pass;
    //atomic_max(minima_value, luma);
}

void kernel init_t0(
    global uint* t0_lattice,
    global uint* t0_labels,
    int width,
    read_only image2d_t gradient_pic,
    global uint* minima_value) {

    uint pixval = read_imageui(gradient_pic, (int2){get_global_id(0), get_global_id(1)}).x;
    uint pos = get_global_id(0)+(get_global_id(1)*width);
    t0_lattice[pos] = pixval <= *minima_value ? (uint)0 : (uint)MAX_INT;
    t0_labels[pos] = pixval <= *minima_value ? (uint)pos : (uint)0;
}

void kernel automaton(
    read_only image2d_t luma_pic, // this contains the values for f(p)
    int width,
    int height,
    int size,
    global const uint* t0_lattice,
    global const uint* t0_labels,
    global uint* t1_lattice,
    global uint* t1_labels) { // a same image cannot be rw

    uint pos = get_global_id(0)+(get_global_id(1)*width);
    // x: north, y: east, z: south, t: west

    uint4 neighborhood_positions = (uint4){
        get_global_id(1) != 0 ? pos-width : pos, // exists if it's not the first row
        get_global_id(0) != (width-1) ? pos+1 : pos, // exists if it's not the last column
        get_global_id(1) != (height-1) ? pos+width : pos, // exists if it's not the last row
        get_global_id(0) != 0 ? pos-1 : pos, //exists if it's not the first column
    };

    uint pixel = read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x;

    uint2 u_t=(uint2){MAX_INT, 9};

    u_t = (neighborhood_positions.x != pos && u_t.x >= t0_lattice[neighborhood_positions.x] + pixel) ?
        (uint2){t0_lattice[neighborhood_positions.x] + pixel, neighborhood_positions.x} : u_t;

    u_t = (neighborhood_positions.y != pos && u_t >= t0_lattice[neighborhood_positions.y] + pixel) ?
        (uint2){t0_lattice[neighborhood_positions.y] + pixel, neighborhood_positions.y} : u_t;

    u_t = (neighborhood_positions.z != pos && u_t >= t0_lattice[neighborhood_positions.z] + pixel) ?
        (uint2){t0_lattice[neighborhood_positions.z] + pixel, neighborhood_positions.z} : u_t;

    u_t = (neighborhood_positions.w != pos && u_t >= t0_lattice[neighborhood_positions.w] + pixel) ?
        (uint2){t0_lattice[neighborhood_positions.w] + pixel, neighborhood_positions.w} : u_t;


    uint u_t_val = u_t.x;
    uint u_t_pos = u_t.y;

    t1_lattice[pos] = t0_lattice[pos] < u_t_val ? t0_lattice[pos] : u_t_val;

    t1_labels[pos] = t0_lattice[pos] < u_t_val ? t0_labels[pos] : t0_labels[u_t_pos];
}

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define R_LUMA_MULT 0.2126f
#define G_LUMA_MULT 0.7152f
#define B_LUMA_MULT 0.0722f
#define MAX_INT 9999999

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

    uint2 u_t=(uint2){MAX_INT, 9};

    u_t = u_t.x >= t0_lattice[neighborhood_positions.x] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x ?
        (uint2){t0_lattice[neighborhood_positions.x] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x, neighborhood_positions.x} : u_t;
    u_t = u_t >= t0_lattice[neighborhood_positions.y] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x ?
        (uint2){t0_lattice[neighborhood_positions.y] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x, neighborhood_positions.y} : u_t;
    u_t = u_t >= t0_lattice[neighborhood_positions.z] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x ?
        (uint2){t0_lattice[neighborhood_positions.z] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x, neighborhood_positions.z} : u_t;
    u_t = u_t >= t0_lattice[neighborhood_positions.w] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x ?
        (uint2){t0_lattice[neighborhood_positions.w] + read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x, neighborhood_positions.w} : u_t;

    uint u_t_val = u_t.x;
    uint u_t_pos = u_t.y;

    t1_lattice[pos] = t0_lattice[pos] < u_t_val ? t0_lattice[pos] : u_t_val;

    t1_labels[pos] = t0_labels[t0_lattice[pos] < u_t_val ? pos : u_t_pos];
}

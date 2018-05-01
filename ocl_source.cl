#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

#define R_LUMA_MULT 0.2126f
#define G_LUMA_MULT 0.7152f
#define B_LUMA_MULT 0.0722f
#define MAX_INT UINT_MAX
#define LAPLACIAN 0

//void kernel init_globals(global uint* minima_value) {
//    *minima_value=255u;
//}

#if LAPLACIAN
constant int ck_gradientx[9] = { // gradient horizontal
        -1,	-1,	-1,
        -1,	 8,	-1,
        -1,	-1,	-1
};
#else
constant int ck_gradientx[9] = { // gradient horizontal
        0,	-1,	0,
        -1,	0,	1,
        0,	1,	0
};
#endif

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

void kernel init_t0(
    global uint* t0_lattice,
    global uint* t0_labels,
    int width,
    read_only image2d_t gradient_pic) {

    uint pixval = read_imageui(gradient_pic, (int2){get_global_id(0), get_global_id(1)}).x;
    uint pos = get_global_id(0)+(get_global_id(1)*width);
    t0_lattice[pos] = pixval == 0 ? (uint)0 : (uint)MAX_INT;
    t0_labels[pos] = pixval == 0 ? (uint)pos : (uint)0;
}

void kernel automaton(
    read_only image2d_t luma_pic, // this contains the values for f(p)
    int width,
    int height,
    global const uint* t0_lattice,
    global const uint* t0_labels,
    global uint* t1_lattice,
    global uint* t1_labels,
    global uint* are_diff) {

    uint pos = get_global_id(0)+(get_global_id(1)*width);
    // x: north, y: east, z: south, t: west

    uint4 neib_pos = (uint4){
        get_global_id(1) != 0 ? pos-width : pos, // exists if it's not the first row
        get_global_id(0) != (width-1) ? pos+1 : pos, // exists if it's not the last column
        get_global_id(1) != (height-1) ? pos+width : pos, // exists if it's not the last row
        get_global_id(0) != 0 ? pos-1 : pos, //exists if it's not the first column
    };

    uint pixel = read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x;

    uint2 u_t=(uint2){
       t0_lattice[pos],
       pos
    };

    uint u_tx = add_sat(t0_lattice[neib_pos.x], (pixel*(pos!=neib_pos.x)));
    uint u_ty = add_sat(t0_lattice[neib_pos.y], (pixel*(pos!=neib_pos.y)));
    uint u_tz = add_sat(t0_lattice[neib_pos.z], (pixel*(pos!=neib_pos.z)));
    uint u_tw = add_sat(t0_lattice[neib_pos.w], (pixel*(pos!=neib_pos.w)));

    u_t = u_t.x > u_tx ? (uint2){u_tx, neib_pos.x} : u_t;
    u_t = u_t.x > u_ty ? (uint2){u_ty, neib_pos.y} : u_t;
    u_t = u_t.x > u_tz ? (uint2){u_tz, neib_pos.z} : u_t;
    u_t = u_t.x > u_tw ? (uint2){u_tw, neib_pos.w} : u_t;

    t1_lattice[pos] = u_t.x;

    t1_labels[pos] = t0_labels[u_t.y];

    atomic_or(&are_diff[0], (
                t0_lattice[pos] != t1_lattice[pos] ||
                t0_labels[pos] != t1_labels[pos]
        )
    );

}

void kernel color_watershed(
    read_only image2d_t original,
    int width,
    int height,
    global const uint* labels,
    write_only image2d_t outimage) {

    int pos = get_global_id(0) + (get_global_id(1) * width); 
    int index = labels[pos];

    uint4 pixel = read_imageui(
        original,
        sampler,
        (int2){
            index % width,
            index / width
        }
    );

    write_imageui(
        outimage,
        (int2){get_global_id(0), get_global_id(1)},
        pixel
    );
}

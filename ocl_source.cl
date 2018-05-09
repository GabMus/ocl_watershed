#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

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
    global uint* are_diff,
    local uint* cache_lattice,
    local uint* cache_labels) {

    const uint pos = get_global_id(0)+(get_global_id(1)*width);
    const uint img_size = width*height;
    const size_t local_id0 = get_local_id(0);
    const size_t local_id1 = get_local_id(1);
    const size_t lws0 = get_local_size(0);
    const size_t lws1 = get_local_size(1);
    const size_t cache_height = lws1+2;
    const int iamoutofbound = get_global_id(0) >= width || get_global_id(1) >= height;

    if (iamoutofbound) return; // failsafe (the global work sizes can be bigger than the image sizes)

    // x: north, y: east, z: south, w: west
    uint4 neib_pos = (uint4){
        get_global_id(1) != 0 ? pos-width : pos, // exists if it's not the first row
        get_global_id(0) != (width-1) ? pos+1 : pos, // exists if it's not the last column
        get_global_id(1) != (height-1) ? pos+width : pos, // exists if it's not the last row
        get_global_id(0) != 0 ? pos-1 : pos //exists if it's not the first column
    };

    // all core indexes are x and y all +1. the formula below linearizes this concept

    uint core_cache_column = local_id0 + 1;
    uint core_cache_row    = local_id1 + 1;
    
    uint local_pos = core_cache_column + (core_cache_row * cache_height);

    // write core pixel in cache
    cache_lattice[local_pos] = t0_lattice[pos];
    cache_labels[local_pos] = t0_labels[pos];

    // Explicit "I am in _ border" declarations as reference. Will do as char4
    // bool i_am_in_north_border = get_local_id(1) == 0;
    // bool i_am_in_east_border = get_local_id(0) == get_local_size(0)-1;
    // bool i_am_in_south_border = get_local_id(1) == get_local_size(0)-1;
    // bool i_am_in_west_border = get_local_id(0) == 0;

    int4 local_border_status = (
	    ((int4){local_id1, local_id0, local_id1, local_id0} ==
	    (int4){0, lws0 - 1, lws1 - 1, 0})
    );

    /*int4 local_border_status = (int4){ 
        get_local_id(1) == 0,                   // North 
        get_local_id(0) == get_local_size(0)-1, // East 
        get_local_id(1) == get_local_size(0)-1, // South 
        get_local_id(0) == 0                    // West 
    };*/

    uint4 local_neib_pos = (uint4){
        local_pos-cache_height,
        local_pos+1,
        local_pos+cache_height,
        local_pos-1
    };
    //local_neib_pos = select((uint4)local_pos, local_neib_pos, local_border_status);

    //if (local_neib_pos.x >= img_size) printf("%u\n", local_neib_pos.x); // TODO: remove

    if (local_border_status.x) {
        cache_lattice[local_neib_pos.x] = t0_lattice[neib_pos.x];
        cache_labels[local_neib_pos.x] = t0_labels[neib_pos.x];
    }
    if (local_border_status.y) {
        cache_lattice[local_neib_pos.y] = t0_lattice[neib_pos.y];
        cache_labels[local_neib_pos.y] = t0_labels[neib_pos.y];
    }
    if (local_border_status.z) {
        cache_lattice[local_neib_pos.z] = t0_lattice[neib_pos.z];
        cache_labels[local_neib_pos.z] = t0_labels[neib_pos.z];
    }
    if (local_border_status.w) {
        cache_lattice[local_neib_pos.w] = t0_lattice[neib_pos.w];
        cache_labels[local_neib_pos.w] = t0_labels[neib_pos.w];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // wait for all work items to finish caching

    //if (iamoutofbound) return;

    uint pixel = read_imageui(luma_pic, (int2){get_global_id(0), get_global_id(1)}).x;

    uint2 u_t=(uint2){
       cache_lattice[local_pos],
       local_pos
    };

    // possible u_t candidates
    uint4 ut_cand = (uint4){
        add_sat(cache_lattice[local_neib_pos.x], pixel),
        add_sat(cache_lattice[local_neib_pos.y], pixel),
        add_sat(cache_lattice[local_neib_pos.z], pixel),
        add_sat(cache_lattice[local_neib_pos.w], pixel),
    };

    ut_cand = select(ut_cand, (uint4)MAX_INT, ((uint4)pos == neib_pos));
    /* the above is a more efficient version of the following
    uint u_tx = pos == neib_pos.x ? MAX_INT : add_sat(cache_lattice[local_neib_pos.x], pixel);
    uint u_ty = pos == neib_pos.y ? MAX_INT : add_sat(cache_lattice[local_neib_pos.y], pixel);
    uint u_tz = pos == neib_pos.z ? MAX_INT : add_sat(cache_lattice[local_neib_pos.z], pixel);
    uint u_tw = pos == neib_pos.w ? MAX_INT : add_sat(cache_lattice[local_neib_pos.w], pixel);*/

    u_t = u_t.x > ut_cand.x ? (uint2){ut_cand.x, local_neib_pos.x} : u_t;
    u_t = u_t.x > ut_cand.y ? (uint2){ut_cand.y, local_neib_pos.y} : u_t;
    u_t = u_t.x > ut_cand.z ? (uint2){ut_cand.z, local_neib_pos.z} : u_t;
    u_t = u_t.x > ut_cand.w ? (uint2){ut_cand.w, local_neib_pos.w} : u_t;

    t1_lattice[pos] = u_t.x;
    uint newlabel = cache_labels[u_t.y];

    t1_labels[pos] = newlabel;

    if (
        cache_lattice[local_pos] != u_t.x || // equivalent to t1_lattice[pos] ||
        cache_labels[local_pos] != newlabel // equivalent to t1_labels[pos]
    ) are_diff[0] = 1;
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

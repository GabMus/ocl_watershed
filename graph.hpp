#pragma once

typedef struct tagGEDGE {
    int voxel1;
    int voxel2;
    int weight;
} *PGEDGE, GEDGE;

typedef struct tagGVOXEX_NEIGHBORHOOD {
    int north;
    int east;
    int south;
    int west;
} *PGVOXEL_NEIGHBORHOOD, GVOXEL_NEIGHBORHOOD;

typedef struct tagRGB {
    int r;
    int g;
    int b;
} *PRGB, RGB;

typedef struct tagGVOXEL {
    int index;
    float value; // luma
    RGB color;
    GVOXEL_NEIGHBORHOOD neighborhood;
} *PGVOXEL, GVOXEL;

void populate_voxel_neighborhood(PGVOXEL voxel, int width, int size) {
    int north = voxel->index - width;
    voxel->neighborhood.north = north >= 0 ? north : -1;
    int south = voxel->index + width;
    voxel->neighborhood.south = south < size ? south : -1;
    voxel->neighborhood.east = ((voxel->index+1) % width) != 0 ? voxel->index + 1 : -1;
    voxel->neighborhood.west = (voxel->index % width) != 0 ? voxel->index -1 : -1;
}

#define R_LUMA_MULT 0.2126f
#define G_LUMA_MULT 0.7152f
#define B_LUMA_MULT 0.0722f

inline float color2luma(RGB color) {
    return ((float)color.r * R_LUMA_MULT)
            + ((float)color.g * G_LUMA_MULT)
            + ((float)color.b * B_LUMA_MULT);
}

void build_vertices(const BMPVEC& image, int width, int height, PGVOXEL* verticesarr) {
    int j=0;
    for (int i=0; i<width*height*3; i+=3) {
        PGVOXEL v;
        v->index = j;
        v->color.r = image[i];
        v->color.g = image[i+1];
        v->color.b = image[i+2];
        v->value = color2luma(v->color);
        populate_voxel_neighborhood(v, width, height);
        verticesarr[j] = v;
        j++;
    }
}

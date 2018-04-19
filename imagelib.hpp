#pragma once
#pragma pack(2)

typedef unsigned int LONG;
typedef unsigned short WORD;
typedef unsigned int DWORD;
typedef unsigned char COLORWORD; // 1 byte
typedef short int COLORPADDING; // 2 byte
typedef std::vector<char> BMPVEC;

typedef struct tagBMPSIZE {
    int w;
    int h;
} *PBMPSIZE, BMPSIZE;

#ifndef std::string
#include <string>
#endif

void read_ppm(std::string path, BMPVEC& buffer, int& width, int& height) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cerr << TERM_RED << "Error: Could not find image for path " << path << std::endl << TERM_RESET;
        exit(1);
    }

    std::string line;

    // P6
    std::getline(file, line);
    // width height
    std::getline(file, line);
    width = std::stoi(line.substr(0, line.find(" ")+1));
    height = std::stoi(line.substr(line.find(" "), line.length()-1));
    // colors
    std::getline(file, line);

    std::streampos offset = file.tellg();

    file.seekg(0,std::ios::end);
    std::streampos length = file.tellg();
    file.seekg(0,std::ios::beg);

    buffer.resize(length);
    file.read(&buffer[0], length);

    BMPVEC::const_iterator first = buffer.begin() + offset;
    BMPVEC::const_iterator last = buffer.end();

    buffer = BMPVEC(first, last);

    file.close();
}

void bgr2bgra(BMPVEC& rawbmp, BMPVEC& bgravec) {
    bgravec.reserve(rawbmp.size() + rawbmp.size()/3);
    int k = 0;
    for (uint i = 0; i < rawbmp.size(); i+=3) {
        bgravec[k] = rawbmp[i];
        bgravec[k+1] = rawbmp[i+1];
        bgravec[k+2] = rawbmp[i+2];
        bgravec[k+3] = 0xFF;
        k+=4;
    }
}

void bgra2rgb(float* invec, int size, unsigned char* outvec) {
    size *= 3;
    unsigned char* p = reinterpret_cast<unsigned char*>(invec);
    int k=0;
    for (int i=0; i<size; i+=3) {
        outvec[size-1-i] = p[k];
        outvec[size-1-(i+1)] = p[k+1];
        outvec[size-1-(i+2)] = p[k+2];
        k+=4;
    }
}

void y_mirror_image(unsigned char* invec, int width, int height, unsigned char* outvec) {
    //width*=3;
    //height*=3;
    for (int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            outvec[width-x-1 + (y*width)] = invec[x + y*width];
        }
    }
}

void r2rgb(unsigned char* invec, int size, unsigned char* outvec) {
    int k=0;
    for (int i=0; i<size; i++) {
        outvec[k] = invec[i];
        outvec[k+1] = invec[i];
        outvec[k+2] = invec[i];
        k+=3;
    }
}

void r32_2_r8(uint32_t* invec, int size, unsigned char* outvec) {
    for (int i=0; i<size; i++) {
        outvec[i] = invec[i] >> 24;
    }
}

bool value_in_arr(uint32_t val, uint32_t* arr, int size) {
    for (int i=0; i<size; i++) {
        if (arr[i] == val) return true;
    }
    return false;
}

uint32_t find_mean_color(int* arr, int size) {
    uint32_t sum=0;
    int pixels=0;
    for (int i=0; i<size; i++) {
        if (arr[i]!=-1) {
            sum+=arr[i];
            pixels++;
        }
    }
    return (uint32_t)(sum/pixels);
}

void color_watershed(uint32_t* labels, uint8_t* image, int width, int height, uint8_t* outimage) {
    for (int x=0; x<width; x++) {
        for (int y=0; y<height; y++) {
            int pos = x + (width * y);
            
#if 0
            if (labels[pos] != pos-1 && labels[pos] != pos) {
                std::cout << "DEBUG: There's a difference!\n";
                std::cout << labels[pos] << " - " << pos << std::endl;
            }
#endif
            //std::cout << "DEBOOG: " << labels[359200] << std::endl;
            //outimage[pos] = !!labels[pos] ? image[labels[pos]-1] : 0;
            outimage[pos] = image[labels[pos]];
        }
    }
}

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <assert.h>

#define PNG_DEBUG 3
#include <png.h>
#include "debug-png-generation.h"

#define CANVAS_WIDTH (12240/120)
#define CANVAS_HEIGHT (12240/120)
#define COLORMAP_HEIGHT 8

/******************************************************************************/
/* Forward declaration of auxiliary, helper functions */
static int f1(int x);
static int f0(int x);
static void numToColor(int64_t num, int64_t max_num, png_byte *pxl_ptr);
static void numToColor_extended(int64_t num, int64_t max_num, png_byte *pxl_ptr);
static void numToColor_simple(int64_t num, int64_t max_num, png_byte *pxl_ptr);
static void init_png_file(void);
static void write_png_file(char *fname);
static void err_exit(const char * s, ...);

/******************************************************************************/
/* Global Variables */
static png_byte bit_depth = 8;
static png_structp png_ptr;
static png_infop info_ptr;
static png_bytep * row_pointers;
static int pxmp_width=CANVAS_WIDTH;
static int pxmp_height=CANVAS_HEIGHT+COLORMAP_HEIGHT;
static int64_t *dague_debug_si_elem_count;
static int debug_png_first_time = 1;

/******************************************************************************/

/* Function bodies */
void dague_pxmp_si_color_rectangle(uint64_t strCol, uint64_t endCol, uint64_t strRow, uint64_t endRow, uint64_t mtrx_height, uint64_t mtrx_width)
{
    uint64_t x,y;
    uint64_t offset;

    if( debug_png_first_time ){
        debug_png_first_time = 0;
        int64_t pixel_count = pxmp_height*pxmp_width;
        dague_debug_si_elem_count = (int64_t *)calloc(pixel_count, sizeof(int64_t));

        init_png_file();

        row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * pxmp_height);
        for (y=0; y<pxmp_height; y++)
                row_pointers[y] = (png_byte*)calloc(1,png_get_rowbytes(png_ptr,info_ptr));

    }

    offset = COLORMAP_HEIGHT*pxmp_width;
    for( y = strRow; y <= endRow; y++){
        for( x = strCol; x <= endCol; x++){
            int64_t py = y*CANVAS_HEIGHT/mtrx_height;
            int64_t px = x*CANVAS_WIDTH/mtrx_width;
            ++dague_debug_si_elem_count[offset + py*pxmp_width + px];
        }
    }

    return;
}

/***/
void dague_pxmp_si_dump_image(char *fname, int64_t max_elem_count)
{
    int64_t max_value = 0;
    int x, y;

    // Maximum possible value.
    max_elem_count /= (CANVAS_WIDTH*CANVAS_HEIGHT);

    for( y = COLORMAP_HEIGHT; y < pxmp_height; y++){
        for( x = 0; x < pxmp_width; x++){
            int64_t tmp = dague_debug_si_elem_count[y*pxmp_width + x];
            if( tmp > max_value )
                max_value = tmp;
        }
    }
    assert(max_value <= max_elem_count);

    //printf("max_value: %ld, max_elem_count: %ld\n",max_value, max_elem_count);

    // paint a stripe showing the color heat-map.
    for( y = 0; y < COLORMAP_HEIGHT-1; y++){
        png_byte* row = row_pointers[y];
        for( x = 0; x < pxmp_width; x++){
            numToColor( x*max_elem_count/pxmp_width, max_elem_count, &(row[x*3]) );
        }
    }
    // Draw a block line to separate the color heat-map from the data.
    {
        y = COLORMAP_HEIGHT-1;
        png_byte* row = row_pointers[y];
        for( x = 0; x < pxmp_width; x++){
            numToColor( max_elem_count, max_elem_count, &(row[x*3]) );
        }
    }
    // paint the actual data
    for( y = COLORMAP_HEIGHT; y < pxmp_height; y++){
        png_byte* row = row_pointers[y];
        for( x = 0; x < pxmp_width; x++){
            numToColor( dague_debug_si_elem_count[y*pxmp_width + x], max_elem_count, &(row[x*3]) );
        }
    }

    write_png_file(fname);
    debug_png_first_time = 1;
}


/******************************************************************************/
static void init_png_file(void)
{
    /* Create the file */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        err_exit("png_create_write_struct failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        err_exit("png_create_info_struct failed");

    /* write the PNG data and meta-data */
    if (setjmp(png_jmpbuf(png_ptr))) err_exit("Error during png_init_io()");

    png_set_IHDR(png_ptr, info_ptr, pxmp_width, pxmp_height,
                 bit_depth, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
}

/***/
void write_png_file(char *fname)
{
    int y;
    FILE *fp = fopen(fname, "w");
    if (!fp) err_exit("File %s could not be opened for writing", fname);

    png_init_io(png_ptr, fp);

    if (setjmp(png_jmpbuf(png_ptr))) err_exit("Error during png_write_info()");

    png_write_info(png_ptr, info_ptr);

    if (setjmp(png_jmpbuf(png_ptr))) err_exit("Error during png_write_image()");

    png_write_image(png_ptr, row_pointers);

    if (setjmp(png_jmpbuf(png_ptr))) err_exit("Error during png_write_end()");

    png_write_end(png_ptr, NULL);

    /* Free the allocated memory */
    for (y=0; y<pxmp_height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    fclose(fp);
}

/***/
static void err_exit(const char * s, ...)
{
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(-1);
}

/** starts high, stays high, falls fast **/
static int f0(int x){
  double f,xd;
  xd = (double)x;
/*  f = 271 -15.9*pow(M_E,(xd/90)); */
  f = 260 -4.53*pow(M_E,(xd/63));
  return((int)f);
}

/** starts low, grows fast, stays high */
static int f1(int x){
  double f,xd;
  xd = (double)x;
  f = 271.0*( 1.0-pow(M_E,-(xd/90.0)) );
  return((int)f);
}

/***/
static void numToColor_middle(int64_t num, int64_t max_num, png_byte *pxl_ptr){
  int k;
  int hue_max_tones = (1<<bit_depth); // 256 for 8 bit color
  int distinct_colors = 4*hue_max_tones; // 1024 for 8 bit color

  num = (int64_t)((double)(distinct_colors-1)*(double)num/(double)max_num);
  k = num/hue_max_tones; // k = {0,1,2,3}
  num = num%hue_max_tones;

  switch(k){
    case 0 : /* white to yellow: drop fast */
             pxl_ptr[0] = (png_byte)(hue_max_tones-1);
             pxl_ptr[1] = (png_byte)(hue_max_tones-1);
             pxl_ptr[2] = (png_byte)(hue_max_tones-1-f1(num));
             break;
    case 1 : /* yellow to green: drop linearly */
             pxl_ptr[0] = (png_byte)(hue_max_tones-1-f1(num)); // drop fast
//             pxl_ptr[0] = (png_byte)(hue_max_tones-1-num); // drop linearly
//             pxl_ptr[0] = (png_byte)f0(num); // drop slowly
             pxl_ptr[1] = (png_byte)(hue_max_tones-1);
             pxl_ptr[2] = (png_byte)0;
             break;
    case 2 : /* green to blue: drop fast and raise fast */
             pxl_ptr[0] = (png_byte)0;
             pxl_ptr[1] = (png_byte)(hue_max_tones-1-f1(num));
             pxl_ptr[2] = (png_byte)f1(num);
             break;
    case 3 : /* blue to black: linear */
             pxl_ptr[0] = (png_byte)0;
             pxl_ptr[1] = (png_byte)0;
             pxl_ptr[2] = (png_byte)(hue_max_tones-1-num);
             break;
    default: fprintf(stderr,"Value is outside color range: %d\n",k);
             exit(-1);
  }

  return;
}

/***/
static void numToColor_extended(int64_t num, int64_t max_num, png_byte *pxl_ptr){
  int k;
  int hue_max_tones = (1<<bit_depth); // 256 for 8 bit color
  int distinct_colors = 6*hue_max_tones; // 1536 for 8 bit color

  // Scale the numbers to speed up the color transition in the beginning.
  num = (int64_t)(sqrt((double)num)*sqrt((double)max_num));

  num = (int64_t)((distinct_colors-1)*(double)num/(double)max_num);
  // WARNING: The following line flips the colormap (from white to yellow to ... to black).
  num = (int64_t)(distinct_colors-1-num);
  k = num/256; // k = {0,1,2,3,4}
  num = num%256;

  pxl_ptr[0] = 0;
  pxl_ptr[1] = 0;
  pxl_ptr[2] = 0;

  switch(k){
    case 0 : /* black to blue */
             pxl_ptr[2] = (png_byte)num;
             break;
    case 1 : /* blue to green */
             pxl_ptr[1] = (png_byte)f1(num);
             pxl_ptr[2] = (png_byte)f0(num);
             break;
    case 2 : /* green to brown */
             pxl_ptr[0] = (png_byte)(f1(num)/2);
             pxl_ptr[1] = (png_byte)((hue_max_tones-1)-3*f1(num)/4);
             break;
    case 3 : /* brown to red */
             pxl_ptr[0] = (png_byte)(hue_max_tones/2+f1(num)/2);
             pxl_ptr[1] = (png_byte)((hue_max_tones-1)/4-f1(num)/4);
             break;
    case 4 : /* red to yellow */
             pxl_ptr[0] = (png_byte)(hue_max_tones-1);
             pxl_ptr[1] = (png_byte)num;
             pxl_ptr[2] = (png_byte)0;
             break;
    case 5 : /* yellow to white: grow slowly and then jump */
             pxl_ptr[0] = (png_byte)(hue_max_tones-1);
             pxl_ptr[1] = (png_byte)(hue_max_tones-1);
             pxl_ptr[2] = (png_byte)(hue_max_tones-1-f0(num));
             break;
    default: fprintf(stderr,"Value is outside color range: %d\n",k);
             exit(-1);
  }
  return;
}

/***/
static void numToColor_simple(int64_t num, int64_t max_num, png_byte *pxl_ptr){
  if( num == 0 ){
      pxl_ptr[0] = 255;
      pxl_ptr[1] = 255;
      pxl_ptr[2] = 255;
      return;
  }
  num = (int64_t)(255*(double)num/((double)max_num));
  num = (int64_t)(255-num);
  pxl_ptr[0] = 0;
  pxl_ptr[1] = 0;
  pxl_ptr[2] = (png_byte)num;

  return;
}

#define EXTENDED_COLORS

static void inline numToColor(int64_t num, int64_t max_num, png_byte *pxl_ptr){
#if defined(SIMPLE_COLORS)
    numToColor_simple(num, max_num, pxl_ptr);
#elif defined(EXTENDED_COLORS)
    numToColor_extended(num, max_num, pxl_ptr);
#else
    numToColor_middle(num, max_num, pxl_ptr);
#endif
}

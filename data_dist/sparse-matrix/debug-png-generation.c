#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#define PNG_DEBUG 3
#include <png.h>
#include "debug-png-generation.h"

/******************************************************************************/
/* Forward declaration of auxiliary, helper functions */
static int f1(int x);
static int f0(int x);
static void numToColor(int64_t num, int64_t max_num, png_byte *pxl_ptr);
static void init_png_file(void);
static void write_png_file(char *fname);
static void err_exit(const char * s, ...);

/******************************************************************************/
/* Global Variables */
static png_byte bit_depth = 8;
static png_structp png_ptr;
static png_infop info_ptr;
static png_bytep * row_pointers;
//static int pxmp_width=12240, pxmp_height=12240;
static int pxmp_width=12240/2, pxmp_height=12240/2;
static int64_t *dague_debug_si_elem_count;

/******************************************************************************/
/* Function bodies */
void dague_pxmp_si_color_rectangle(uint64_t strCol, uint64_t endCol, uint64_t strRow, uint64_t endRow, uint64_t mtrx_height, uint64_t mtrx_width)
{
    uint32_t x,y;
    static int first_time = 1;

    if( first_time ){
        first_time = 0;
        dague_debug_si_elem_count = (int64_t *)calloc(pxmp_height*pxmp_width, sizeof(int64_t));

        init_png_file();

        row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * pxmp_height);
        for (y=0; y<pxmp_height; y++)
                row_pointers[y] = (png_byte*)calloc(1,png_get_rowbytes(png_ptr,info_ptr));

    }

    for( y = (pxmp_height*strRow)/mtrx_height; y <= (pxmp_height*endRow)/mtrx_height; y++){
        for( x = (pxmp_width*strCol)/mtrx_width; x <= (pxmp_width*endCol)/mtrx_width; x++){
            ++dague_debug_si_elem_count[y*pxmp_width + x];
        }
    }
}

/***/
void dague_pxmp_si_dump_image(char *fname, int64_t max_elem_count)
{
    int64_t max_value = 0;
    int x, y;

    for( y = 0; y < pxmp_height; y++){
        for( x = 0; x < pxmp_width; x++){
            int64_t tmp = dague_debug_si_elem_count[y*pxmp_width + x];
            if( tmp > max_value )
                max_value = tmp;
        }
    }

    max_elem_count /= (pxmp_width*pxmp_height);

    printf("max_value: %ld, max_elem_count: %ld\n",max_value, max_elem_count);

    for( y = 0; y < pxmp_height; y++){
        png_byte* row = row_pointers[y];
        for( x = 0; x < pxmp_width; x++){
//            numToColor( dague_debug_si_elem_count[y*pxmp_width + x], max_value, &(row[x*3]) );
            numToColor( dague_debug_si_elem_count[y*pxmp_width + x], max_elem_count, &(row[x*3]) );
        }
    }

    write_png_file(fname);
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

/***/
static int f0(int x){
  double f,xd;
  xd = (double)x;
/*  f = 271 -15.9*pow(M_E,(xd/90)); */
  f = 260 -4.53*pow(M_E,(xd/63));
  return((int)f);
}

/***/
static int f1(int x){
  double f,xd;
  xd = (double)x;
  f = 271.0*( 1.0-pow(M_E,-(xd/90.0)) );
  return((int)f);
}

/***/
static void numToColor(int64_t num, int64_t max_num, png_byte *pxl_ptr){
  int k;

/* simplified version */
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
  pxl_ptr[2] = num;

  return;

/* extended, but weird heat map */

  num = (int64_t)(1280*(double)num/((double)max_num+1));
  num = (int64_t)(1280-num-1);
  k = num/256; // k = {0,1,2,3,4}
  num = num%256;
// We don't need to discretize num, we like the colors smooth
// num = 15*(num/15); // integer division 

  pxl_ptr[0] = 0;
  pxl_ptr[1] = 0;
  pxl_ptr[2] = 0;

  switch(k){
    case 0 : /* black to blue */
             pxl_ptr[2] = num;
             break;
    case 1 : /* blue to green */
             pxl_ptr[1] = f1(num);
             pxl_ptr[2] = f0(num);
             break;
    case 2 : /* green to brown */
             pxl_ptr[0] = num;
             pxl_ptr[1] = (1<<bit_depth)-1;
             break;
    case 3 : /* brown to red */
             pxl_ptr[0] = (1<<bit_depth)-1; 
             pxl_ptr[1] = f0(num);
             break;
    case 4 : /* red to white */
             pxl_ptr[0] = (1<<bit_depth)-1; 
             pxl_ptr[1] = num;
             pxl_ptr[2] = num;
             break;
    default: fprintf(stderr,"Impossible value: %d\n",k);
             exit(-1);
  }
  return;
}

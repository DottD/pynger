// StdLib
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
// NBIS
#ifdef __linux__
#define current_time unused_function
#endif
#include <pca.h>
#ifdef __linux__
#undef current_time
#endif
// #define PYNGER_DEBUG

#ifdef __linux__
#include <time.h>
char *current_time(void);
#endif

void flatten_2d_array_b(unsigned char** flattened, unsigned char** array2d, 
	const size_t width, const size_t height, const bool realloc);
		
void flatten_2d_array_c(signed char** flattened, signed char** array2d, 
	const size_t width, const size_t height, const bool realloc);
	
void flatten_2d_array_f(float** flattened, float** array2d, 
	const size_t width, const size_t height, const bool realloc);

void build_2d_array_b(unsigned char*** array2d, unsigned char* flattened, 
	const size_t width, const size_t height, const bool realloc);

void build_2d_array_c(signed char*** array2d, signed char* flattened, 
	const size_t width, const size_t height, const bool realloc);

void build_2d_array_f(float*** array2d, float* flattened, 
	const size_t width, const size_t height, const bool realloc);
	
#ifdef PYNGER_DEBUG
void print_2d_array_char(char** array2d, const size_t width, const size_t height);

void print_2d_array_uchar(unsigned char** array2d, const size_t width, const size_t height);

void print_2d_array_flt(float** array2d, const size_t width, const size_t height);

void dbg_find_abnormal_char_2d(char** array2d, const size_t width, const size_t height);

void dbg_find_abnormal_char(char* array2d, const size_t length);

void dbg_find_abnormal_uchar_2d(unsigned char** array2d, const size_t width, const size_t height);

void dbg_find_abnormal_uchar(unsigned char* array2d, const size_t length);

void dbg_find_abnormal_flt_2d(float** array2d, const size_t width, const size_t height);

void dbg_find_abnormal_flt(float* array2d, const size_t length);

void print_array_1d(const unsigned char*, const int, const int);
void print_array_2d(const unsigned char**, const int, const int);
#endif
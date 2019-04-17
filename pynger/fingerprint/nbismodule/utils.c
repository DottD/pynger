#include "utils.h"

/* Convenience function that handles unsigned char array */
/* Note: allocates the new array */
void flatten_2d_array_b(unsigned char** flattened, unsigned char** array2d, 
	const size_t width, const size_t height, const bool realloc)
{
	if (realloc){
		malloc_uchar(flattened, width*height, "utils.c -> flatten_2d_array_b -> flattened");
	}
	unsigned char* flat_ptr = *flattened;
	size_t row_byte_width = sizeof(unsigned char) * width;
	for(size_t i = 0; i < height; ++i)
	{
		memcpy(flat_ptr, array2d[i], row_byte_width);
		flat_ptr += width;
	}
}

/* Convenience function that handles char array */
/* Note: allocates the new array */
void flatten_2d_array_c(signed char** flattened, signed char** array2d, 
	const size_t width, const size_t height, const bool realloc)
{
	if (realloc){
		malloc_char((char**)flattened, width*height, "utils.c -> flatten_2d_array_c -> flattened");
	}
	signed char* flat_ptr = *flattened;
	size_t row_byte_width = sizeof(signed char) * width;
	for(size_t i = 0; i < height; ++i)
	{
		memcpy(flat_ptr, array2d[i], row_byte_width);
		flat_ptr += width;
	}
}	

/* Convenience function that handles float array */
/* Note: allocates the new array */
void flatten_2d_array_f(float** flattened, float** array2d, 
	const size_t width, const size_t height, const bool realloc)
{
	if (realloc){
		malloc_flt(flattened, width*height, "utils.c -> flatten_2d_array_f -> flattened");
	}
	float* flat_ptr = *flattened;
	size_t row_byte_width = sizeof(float) * width;
	for(size_t i = 0; i < height; ++i)
	{
		memcpy(flat_ptr, array2d[i], row_byte_width);
		flat_ptr += width;
	}
}

/* Convenience function that handles unsigned char array */
/* Note: allocates the new array */
void build_2d_array_b(unsigned char*** array2d, unsigned char* flattened, 
	const size_t width, const size_t height, const bool realloc)
{
	if (realloc){
		malloc_dbl_uchar(array2d, height, width, "utils.c -> build_2d_array_b -> array2d");
	}
	unsigned char** struct_ptr = *array2d;
	size_t row_byte_size = sizeof(unsigned char) * width;
	for(size_t i = 0; i < height; ++i)
	{
		memcpy(struct_ptr[i], flattened, row_byte_size);
		flattened += width;
	}
}

/* Convenience function that handles signed char array */
/* Note: allocates the new array */
void build_2d_array_c(signed char*** array2d, signed char* flattened, 
	const size_t width, const size_t height, const bool realloc)
{
	if (realloc){
		malloc_dbl_char((char***)array2d, height, width, "utils.c -> build_2d_array_c -> array2d");
	}
	char** struct_ptr = *array2d;
	size_t row_byte_size = sizeof(char) * width;
	for(size_t i = 0; i < height; ++i)
	{
		memcpy(struct_ptr[i], flattened, row_byte_size);
		flattened += width;
	}
}

/* Convenience function that handles float array */
/* Note: allocates the new array */
void build_2d_array_f(float*** array2d, float* flattened, 
	const size_t width, const size_t height, const bool realloc)
{
	if (realloc){
		malloc_dbl_flt(array2d, height, width, "utils.c -> build_2d_array_f -> array2d");
	}
	float** struct_ptr = *array2d;
	size_t row_byte_size = sizeof(float) * width;
	for(size_t i = 0; i < height; ++i)
	{
		memcpy(struct_ptr[i], flattened, row_byte_size);
		flattened += width;
	}
}

#ifdef PYNGER_DEBUG
/* Print to stdout the values of the given 2D array */
void print_2d_array_char(char** array2d, const size_t width, const size_t height)
{
	for(size_t i = 0; i < height; ++i)
	{
		for(size_t j = 0; j < width; ++j)
		{
			printf("%d ", array2d[i][j]);
		}
		printf("\n");
	}
}

/* Print to stdout the values of the given 2D array */
void print_2d_array_uchar(unsigned char** array2d, const size_t width, const size_t height)
{
	for(size_t i = 0; i < height; ++i)
	{
		for(size_t j = 0; j < width; ++j)
		{
			printf("%u ", array2d[i][j]);
		}
		printf("\n");
	}
}

/* Print to stdout the values of the given 2D array */
void print_2d_array_flt(float** array2d, const size_t width, const size_t height)
{
	for(size_t i = 0; i < height; ++i)
	{
		for(size_t j = 0; j < width; ++j)
		{
			printf("%f ", array2d[i][j]);
		}
		printf("\n");
	}
}


/* Print to stdout the location and value of abnormal values (i.e. nan and inf) */
void dbg_find_abnormal_char_2d(char** array2d, const size_t width, const size_t height)
{
	for(size_t i = 0; i < height; ++i)
	{
		for(size_t j = 0; j < width; ++j)
		{
			if(!isnormal(array2d[i][j]))
			{
				printf("Abnormal value %d found at (%zu, %zu)\n", array2d[i][j], i, j);
			}
		}
	}
}

void dbg_find_abnormal_char(char* array, const size_t length)
{
	for(size_t k = 0; k < length; ++k)
	{
		if(!isnormal(array[k]))
		{
			printf("Abnormal value %d found at %zu-th position\n", array[k], k);
		}
	}
}

void dbg_find_abnormal_uchar_2d(unsigned char** array2d, const size_t width, const size_t height)
{
	for(size_t i = 0; i < height; ++i)
	{
		for(size_t j = 0; j < width; ++j)
		{
			if(!isnormal(array2d[i][j]))
			{
				printf("Abnormal value %u found at (%zu, %zu)\n", array2d[i][j], i, j);
			}
		}
	}
}

void dbg_find_abnormal_uchar(unsigned char* array, const size_t length)
{
	for(size_t k = 0; k < length; ++k)
	{
		if(!isnormal(array[k]))
		{
			printf("Abnormal value %u found at %zu-th position\n", array[k], k);
		}
	}
}

void dbg_find_abnormal_flt_2d(float** array2d, const size_t width, const size_t height)
{
	for(size_t i = 0; i < height; ++i)
	{
		for(size_t j = 0; j < width; ++j)
		{
			if(!isnormal(array2d[i][j]))
			{
				printf("Abnormal value %f found at (%zu, %zu)\n", array2d[i][j], i, j);
			}
		}
	}
}

void dbg_find_abnormal_flt(float* array, const size_t length)
{
	for(size_t k = 0; k < length; ++k)
	{
		if(!isnormal(array[k]))
		{
			printf("Abnormal value %f found at %zu-th position\n", array[k], k);
		}
	}
}

void print_array_1d(const unsigned char *p, const int nr, const int nc){
	FILE *fp = fopen ("/Users/MacD/Download/sgmnt_dump.txt", "w+");
	fprintf(fp, "\nnp.array([");
	for(int i = 0; i < nr*nc; ++i){
		fprintf(fp, "%d, ", p[i]);
	}
	fprintf(fp, "]).reshape((%d, %d))\n", nr, nc);
   	fclose(fp);
}
void print_array_2d(const unsigned char **p, const int nr, const int nc){
	FILE *fp = fopen ("/Users/MacD/Download/sgmnt_dump.txt", "w+");
	fprintf(fp, "\nnp.array([");
	for(int i = 0; i < nr; ++i){
		for(int j = 0; j < nc; ++j){
			fprintf(fp, "%d, ", p[i][j]);
		}
	}
	fprintf(fp, "]).reshape((%d, %d))\n", nr, nc);
   	fclose(fp);
}
#endif
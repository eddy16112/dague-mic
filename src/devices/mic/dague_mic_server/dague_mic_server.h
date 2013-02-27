#ifndef DAGUE_MIC_SERVER_H
#define DAGUE_MIC_SERVER_H

#include <scif.h>

#define SUCCESS			1
#define ERROR			0

#define PAGE_SIZE       0x1000

#define COMMAND_FUN_MAX     100

#define MALLOC_MEM			10
#define CLOSE_MIC			11
#define SYNC_WRITE			12
#define MIC_DGEMM			20

enum micMemcpyKind {
	micMemcpyHostToHost,
	micMemcpyHostToDevice,
	micMemcpyDeviceToHost,
	micMemcpyDeviceToDevice
};

typedef struct mic_mem_struct {
	char *addr;
	off_t offset;
	size_t actual_nbyte;
}mic_mem_t;

typedef struct mic_gemm_param {
	mic_mem_t gemm_A;
	mic_mem_t gemm_B;
	mic_mem_t gemm_C;
	int m;
	int n;
	int k;
}mic_gemm_param_t;

typedef int micEvent_t;

typedef int (*mic_command_function_t)();

/** send and receive command to/from client */
int mic_send_sync(scif_epd_t epd, void *msg, int len);
int mic_recv_sync(scif_epd_t epd, void *msg, int len);

/** malloc mem in mic */
int mic_malloc_server(void **addr, size_t nbyte, off_t *offset);

/** process command from client */
int mic_command_array_init();
int mic_server_process_command(mic_command_function_t fp);
int mic_command_malloc();

/** init and close server */
int mic_server_init();
int mic_server_close();

#endif

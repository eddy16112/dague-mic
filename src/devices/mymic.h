/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _MYMIC_H_
#define _MYMIC_H_

#include <scif.h>

#define MALLOC_MEM			100
#define MIC_SUCCESS			1
#define MIC_ERROR			0

typedef struct mic_mem_struct {
	void *addr;
	off_t offset;
	size_t actual_nbyte;
}mic_mem_t;

static inline int micMalloc(mic_mem_t *mic_mem_dev, size_t size);
static inline int mic_send_sync(scif_epd_t epd, void *msg, int len);
static inline int mic_recv_sync(scif_epd_t epd, void *msg, int len);
static scif_epd_t epd;


int micMalloc(mic_mem_t *mic_mem_dev, size_t size)
{
	int command = MALLOC_MEM; 
	if (mic_send_sync(epd, &command, sizeof(command)) == MIC_ERROR) { 
		return MIC_ERROR;
	}
	//make size multiple of PAGE_SIZE                                                                             
  /*	if (size % PAGE_SIZE != 0) {
    	size = ((size / PAGE_SIZE) + 1) * PAGE_SIZE;
  	}*/
	if (mic_send_sync(epd, &size, sizeof(size)) == MIC_ERROR) { 
		return MIC_ERROR;
	}
	printf("%lu\n", size);
	if (mic_recv_sync(epd, mic_mem_dev, sizeof(mic_mem_t)) == MIC_ERROR) { 
		return MIC_ERROR;
	}
	printf("malloc addr after recv:%p offset 0x%lx nbyte %lu\n", mic_mem_dev->addr, mic_mem_dev->offset, mic_mem_dev->actual_nbyte);
	return MIC_SUCCESS;
}

int mic_send_sync(scif_epd_t epd, void *msg, int len) {
	int err;
	if ((err = scif_send(epd, msg, len, 1)) <= 0) {
		err = errno;
		printf("scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return MIC_ERROR;
	}
	return MIC_SUCCESS;
}

int mic_recv_sync(scif_epd_t epd, void *msg, int len) {
	int err;
	if ((err = scif_recv(epd, msg, len, 1)) <= 0) {
		err = errno;
		printf("scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return MIC_ERROR;
	}
	return MIC_SUCCESS;
}

#endif /* _MYMIC_H_ */

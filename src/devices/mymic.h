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

#define PAGE_SIZE       0x1000

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

static inline int micMalloc(mic_mem_t *mic_mem_dev, size_t size);
static inline int mic_send_sync(scif_epd_t epd, void *msg, int len);
static inline int mic_recv_sync(scif_epd_t epd, void *msg, int len);
static inline int micHostAlloc(mic_mem_t *mic_mem_host, size_t size);
static inline int micInit();
static inline int micMemcpyAsync(void* host_addr, off_t roffset, size_t length, enum micMemcpyKind kind);
static inline int micDeviceGetCount(int * count);
static inline int micMemGetInfo(size_t *free_mem, size_t *total_mem );
static inline mic_mem_t * micInitEventQueue(int size);
static inline int micEventRecord(mic_mem_t *event_queue, int event_no, int stream);
static inline int micEventQuery(mic_mem_t *event_queue, int event_no);
static scif_epd_t epd;


static inline int micMalloc(mic_mem_t *mic_mem_dev, size_t size)
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

static inline int mic_send_sync(scif_epd_t epd, void *msg, int len) {
	int err;
	if ((err = scif_send(epd, msg, len, 1)) <= 0) {
		err = errno;
		printf("scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return MIC_ERROR;
	}
	return MIC_SUCCESS;
}

static inline int mic_recv_sync(scif_epd_t epd, void *msg, int len) {
	int err;
	if ((err = scif_recv(epd, msg, len, 1)) <= 0) {
		err = errno;
		printf("scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return MIC_ERROR;
	}
	return MIC_SUCCESS;
}

static inline int micHostAlloc(mic_mem_t *mic_mem_host, size_t size)
{
	void *pHost;
	off_t offset, suggested_offset;
    
	//make size multiple of PAGE_SIZE
  	if (size % PAGE_SIZE != 0) {
    	size = ((size / PAGE_SIZE) + 1) * PAGE_SIZE;
  	}
	posix_memalign(&pHost, PAGE_SIZE, size);
	if (pHost == NULL) {
		return MIC_ERROR;
	}
	suggested_offset = 0x0;
	if((offset = scif_register(epd, pHost, size, suggested_offset,
                               SCIF_PROT_READ | SCIF_PROT_WRITE, 0)) < 0) {
    	printf("scif_register failed with err %d\n", errno);
    	fflush(stdout);
    	return MIC_ERROR;
  	}
	mic_mem_host->addr = pHost;
	mic_mem_host->offset = offset;
	mic_mem_host->actual_nbyte = size;
	printf("malloc host:%p offset 0x%lx nbyte %lu\n", mic_mem_host->addr, mic_mem_host->offset, mic_mem_host->actual_nbyte);
	return MIC_SUCCESS;
	
}

static inline int micInit()
{
	int conn_port, req_port, conn, tries;
	struct scif_portID portID;
    
	req_port = 2049;
	portID.node = 1;
    portID.port = 2050;
    
	printf("Client is trying to connect MIC server ...\n");
    
	/* create a end point, return a descriptor */
	if ((epd = scif_open()) < 0) {
		printf("scif_open failed with error %d\n", (int)epd);
		return MIC_ERROR;
	}
    
	/* bind port */
	if ((conn_port = scif_bind(epd, req_port)) < 0) {
		printf("scif_bind failed with error %d\n", conn_port);
		return MIC_ERROR;
	}
	printf("Bind to port %d success\n", conn_port);
    
	tries = 5;
	conn = scif_connect(epd, &portID);
	while (conn < 0) {
		if ((errno == ECONNREFUSED) && (tries > 0)) {
			printf("Connection to node %d failed : trial %d\n", portID.node, tries);
			tries--;
			sleep(1);
		}
		else {
			printf("Connection failed with error %d\n", errno);
			return MIC_ERROR;
		}
		conn = scif_connect(epd, &portID);
	}
	printf("Conection established, connect to node %d success\n", portID.node);
    
	return MIC_SUCCESS;
}

static inline int micMemcpyAsync(void* host_addr, off_t roffset, size_t length, enum micMemcpyKind kind)
{
	int err = 0;
	double *value;
	//printf("src size %lu, dst size %lu\n", src->actual_nbyte, dst->actual_nbyte);

	if (kind == micMemcpyHostToDevice) {
//		printf("set value %f, diff %lu\n", value[512], diff);
		if ((err = scif_vwriteto(epd,
					host_addr, /* local RAS offset */
					length,
					roffset, /* remote RAS offset */
					0))) {

			printf("scif_writeto failed with err %d\n", errno);
			return MIC_ERROR;
		
		}
		
	} else if (kind == micMemcpyDeviceToHost) {
	//	printf("before get value %f\n", value[4000000]);
		if ((err = scif_vreadfrom(epd,
					host_addr, /* local RAS offset */
					length,
					roffset, /* remote RAS offset */
					0))) {

			printf("scif_writeto failed with err %d\n", errno);
			return MIC_ERROR;
		}
	//	sleep(5);
	//	printf("get value %f\n", value[79]);
	}

	return MIC_SUCCESS;
}

static inline int micMemGetInfo(size_t *free_mem, size_t *total_mem )
{
	*free_mem = sizeof(double)*4096*1024*64;
	*total_mem = sizeof(double)*4096*1024*64;
	return MIC_SUCCESS;
}

static inline int micDeviceGetCount(int * count)
{
	*count = 1;
	return MIC_SUCCESS;
}

static inline mic_mem_t * micInitEventQueue(int size)
{
	int i, rc;
	uint64_t *value;

	mic_mem_t *event_queue = (mic_mem_t *)malloc(sizeof(mic_mem_t));
	rc = micHostAlloc(event_queue, size);
	if (rc != MIC_SUCCESS) {
		return NULL;
	}

	value = (uint64_t *)event_queue->addr;
	for (i = 0; i < size; i++) {
		value[i] = 0;
	}
	return event_queue;
}

static inline int micEventRecord(mic_mem_t *event_queue, int event_no, int stream)
{
	int i = 1, rc;
	uint64_t v;
	if (i) {    // for memcpy only, but I dont know how to check the task type now.
		off_t base_offset, now_offset;
		base_offset = event_queue->offset;
		size_t diff = sizeof(uint64_t)*event_no;
		now_offset = base_offset + diff;
		v = 1;
		rc = scif_fence_signal (epd, now_offset, v, 0, 0, SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_LOCAL);
		 
	} else {    // for compute
	}
	return MIC_SUCCESS;
}

static inline int micEventQuery(mic_mem_t *event_queue, int event_no)
{
	uint64_t *queue_value = (uint64_t *)event_queue->addr;
	if (queue_value[event_no] == 1) {
		queue_value[event_no] = 0;
		return MIC_SUCCESS;
	} else {
		return MIC_ERROR;
	}
}

#endif /* _MYMIC_H_ */

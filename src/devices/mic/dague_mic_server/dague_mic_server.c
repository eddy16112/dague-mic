#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <mkl.h>
#include "dague_mic_server.h"

scif_epd_t epd;
scif_epd_t mic_epd;

mic_command_function_t mic_command_array[COMMAND_FUN_MAX];

int mic_send_sync(scif_epd_t epd, void *msg, int len) {
	int err;
	if ((err = scif_send(epd, msg, len, 1)) <= 0) {
		err = errno;
		printf("[Error] scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return ERROR;
	}
	return SUCCESS;
}

int mic_recv_sync(scif_epd_t epd, void *msg, int len) {
	int err;
	if ((err = scif_recv(epd, msg, len, 1)) <= 0) {
		err = errno;
		printf("[Error] scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return ERROR;
	}
	return SUCCESS;
}

int mic_malloc_server(void **addr, size_t nbyte, off_t *offset)
{
	off_t suggested_offset;
 	int err = 0;
	*addr = mmap( NULL, nbyte, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0 );
	if (addr == MAP_FAILED || *addr == NULL) {
		printf("[Error] mmap  failed %d\n", err);
		return ERROR;
	}
//	posix_memalign(&addr, PAGE_SIZE, nbyte);

	suggested_offset = 0x0;

	if((*offset = scif_register(mic_epd, *addr, nbyte, suggested_offset,
				SCIF_PROT_READ | SCIF_PROT_WRITE, 0)) < 0) {
		printf("[Error] scif_register failed with err %d\n", errno);
     //   check_error_code((int)errno);
		fflush(stdout);
		return ERROR;
	}
	printf("inside malloc %p, offset 0x%lx \n", *addr, *offset);
	//printf("mic_malloc: 0x%lx offset: 0x%lx nbyte: %ld pagesize: %ld\n", address, *offset, nbyte, (size_t)sysconf(_SC_PAGESIZE));
	//suggested_offset += *offset;

	return SUCCESS;
}

int mic_server_init()
{
	int req_port, conn_port, req_queue_len;
	struct scif_portID portID;
	
	req_port = 2055;
	req_queue_len = 5;

	printf("[Log] Server starting...\n");

	/* create a end point, return a descriptor */
	if ((epd = scif_open()) < 0) {
		printf("[Error] scif_open failed with error %d\n", (int)epd);
		return ERROR;
	}

	/* bind port */
	if ((conn_port = scif_bind(epd, req_port)) < 0) {
		printf("[Error] scif_bind failed with error %d\n", conn_port);
		return ERROR;
	}
	printf("[Log] Bind to port %d success\n", conn_port);

	 /* req_queue_len is length of request queue */
	if (scif_listen(epd, req_queue_len) != 0) {
		printf("[Error] scif_listen failed with error %d\n", errno);
		return ERROR;
	}
	printf("[Log] Server starts successfully, waiting for request...\n");

	if (scif_accept(epd, &portID, &mic_epd, SCIF_ACCEPT_SYNC) != 0) {
		printf("[Error] scif_accept failed with error %d\n", errno);
		return ERROR;
	}
	printf("[Log] Connection established, accepted request from node:%d port:%d\n", portID.node, portID.port);

	return SUCCESS;
}

int mic_server_close()
{
	if ((scif_close(epd) != 0) || (scif_close(mic_epd) != 0)) {
		printf("[Error] scif_close failed with error %d\n", errno);
		return ERROR;
	}
	printf("[Log] Server closed success\n");
	return SUCCESS;

}

int mic_command_malloc()
{
    off_t offset;
	mic_mem_t mic_mem_info;
    size_t nbyte;
    void *addr = NULL;
    
    printf ("[Log] malloc command acceptted\n");
    if (mic_recv_sync(mic_epd, &nbyte, sizeof(nbyte)) == ERROR) {
        return ERROR;
    }
    printf("nbyte %lu\n", nbyte);
    if(mic_malloc_server(&addr, nbyte, &offset) == ERROR) {
        return ERROR;
    }
    mic_mem_info.addr = addr;
    mic_mem_info.offset = offset;
    mic_mem_info.actual_nbyte = nbyte;
    printf("malloc addr before send:%p offset 0x%lx\n", mic_mem_info.addr, mic_mem_info.offset);
    if (mic_send_sync(mic_epd, &mic_mem_info, sizeof(mic_mem_info)) == ERROR) {
        return ERROR;
    }
    printf("[Log] finish malloc command\n");
    return SUCCESS;
}

int mic_server_process_command(mic_command_function_t fp)
{
    fp();
    return SUCCESS;
}

int process_task()
{
	int command, run_flag;
	run_flag = 1;

	while (run_flag) {
		if (mic_recv_sync(mic_epd, &command, sizeof(command)) == ERROR) { 
			return ERROR;
		}
        
		if (command >= COMMAND_FUN_MAX || mic_command_array[command] == NULL) {
            printf("unknow command %d\n", command);
        } else if (command == CLOSE_MIC) {
            run_flag = 0;
        } else {
            mic_server_process_command(mic_command_array[command]);
        }
	}

	

	return SUCCESS;

}

int mic_command_array_init()
{
    int i;
    for (i = 0; i < COMMAND_FUN_MAX; i++) {
        mic_command_array[i] = NULL;
    }
    mic_command_array[MALLOC_MEM] = mic_command_malloc;
  //  mic_command_array[MIC_DGEMM] = mic_command_dgemm;
    return SUCCESS;
}

int main(int argc, char *argv[])
{
	if (mic_server_init() != SUCCESS) {
		printf("[Error]:server can not start\n");
		exit(1);
	}
    mic_command_array_init();
	process_task();	

	if (mic_server_close() != SUCCESS) {
		printf("[Error]:server can not close\n");
		exit(1);
	}
	return 0;	
}

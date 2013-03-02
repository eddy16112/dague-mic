#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <mkl.h>
#include "dague_mic_server.h"

#define MIC_TASK_NUM_OF_THREADS    4

scif_epd_t epd;
scif_epd_t mic_epd;

mic_command_function_t mic_command_array[COMMAND_FUN_MAX];
mic_task_function_t mic_task_array[COMMAND_FUN_MAX];
mic_task_queue_t *mic_server_task_queue;

int mic_send_sync(scif_epd_t mepd, void *msg, int len) {
	int err;
	if ((err = scif_send(mepd, msg, len, 1)) <= 0) {
		err = errno;
		printf("[Error] scif_send failed with err %d\n", errno);
		fflush(stdout);
		return ERROR;
	}
	return SUCCESS;
}

int mic_send_async(scif_epd_t mepd, void *msg, int len) {
	int err;
	if ((err = scif_send(mepd, msg, len, 0)) <= 0) {
		err = errno;
		printf("[Error] scif_send failed with err %d\n", errno);
		fflush(stdout);
		return ERROR;
	}
	return SUCCESS;
}

int mic_recv_sync(scif_epd_t mepd, void *msg, int len) {
	int err;
	if ((err = scif_recv(mepd, msg, len, 1)) <= 0) {
		err = errno;
		printf("[Error] scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return ERROR;
	}
	return SUCCESS;
}

int mic_recv_async(scif_epd_t mepd, void *msg, int len) {
	int err;
	if ((err = scif_recv(mepd, msg, len, 0)) < 0) {
		err = errno;
		printf("[Error] scif_recv failed with err %d\n", errno);
		fflush(stdout);
		return err;
	}
	return err;
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
	req_queue_len = 25;

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

int mic_server_task_enqueue(mic_task_context_t * this_task)
{
    pthread_mutex_lock(&(mic_server_task_queue->task_queue_lock));
    if (mic_server_task_queue->total_task == 0) {
        mic_server_task_queue->head = this_task;
        mic_server_task_queue->execution = this_task;
    } else if (mic_server_task_queue->runable_task == 0) {
        mic_server_task_queue->execution = this_task;
        mic_server_task_queue->end->next_task = this_task;
    } else {
        mic_server_task_queue->end->next_task = this_task;
    }
    mic_server_task_queue->end = this_task;
    mic_server_task_queue->total_task ++;
    mic_server_task_queue->runable_task ++;
    this_task->task_status = TASK_READY;
    printf("[Log] task %p enqueued, id %d, total task: %d, runable task: %d\n", this_task, this_task->task_id, mic_server_task_queue->total_task, mic_server_task_queue->runable_task);
    pthread_mutex_unlock(&(mic_server_task_queue->task_queue_lock));

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

int mic_command_dgemm()
{
    mic_task_context_t *this_task;

    printf("[Log] dgemm command accepted\n");
    mic_gemm_param_t *mic_dgemm_param = (mic_gemm_param_t *)malloc(sizeof(mic_gemm_param_t));
    if (mic_recv_sync(mic_epd, mic_dgemm_param, sizeof(mic_gemm_param_t)) == ERROR) { 
        return ERROR;
    }
    this_task = (mic_task_context_t *)malloc(sizeof(mic_task_context_t));
    this_task->task_name = MIC_DGEMM;
    this_task->task_data = mic_dgemm_param;
    this_task->task_id = mic_dgemm_param->task_id;
    this_task->next_task = NULL;

    mic_server_task_enqueue(this_task);

    return SUCCESS;
	
}

int mic_command_sgemm()
{
    mic_task_context_t *this_task;

    printf("[Log] sgemm command accepted\n");
    mic_gemm_param_t *mic_sgemm_param = (mic_gemm_param_t *)malloc(sizeof(mic_gemm_param_t));
    if (mic_recv_sync(mic_epd, mic_sgemm_param, sizeof(mic_gemm_param_t)) == ERROR) { 
        return ERROR;
    }
    this_task = (mic_task_context_t *)malloc(sizeof(mic_task_context_t));
    this_task->task_name = MIC_DGEMM;
    this_task->task_data = mic_sgemm_param;
    this_task->task_id = mic_sgemm_param->task_id;
    this_task->next_task = NULL;

    mic_server_task_enqueue(this_task);

    return SUCCESS;
	
}

int mic_command_check_completion()
{
    mic_task_context_t *this_task;
    int ack = -1;
    
  //  printf("[Log] check completion accepted\n");
    this_task = mic_server_task_queue->head;
    if (this_task != NULL) {
    //    printf("i am herer\n");
        if (this_task->task_status == TASK_FINISH) {
            pthread_mutex_lock(&(mic_server_task_queue->task_queue_lock));
            mic_server_task_queue->head = this_task->next_task;
            mic_server_task_queue->total_task --;
            pthread_mutex_unlock(&(mic_server_task_queue->task_queue_lock));
            ack = 1;
            if (mic_send_sync(mic_epd, &ack, sizeof(ack)) == ERROR) {
                return ERROR;
            }
            if (this_task->task_data != NULL) {
                free(this_task->task_data);
            }
            ack = this_task->task_id;
            printf("I send task %d ACK back, this %p, next %p\n", ack, this_task, this_task->next_task);
            free(this_task);
            return SUCCESS;
        } else {
            ack = 0;
            if (mic_send_sync(mic_epd, &ack, sizeof(ack)) == ERROR) {
                return ERROR;
            }
        }
    }
    ack = 2;     // task queue is empty
    if (mic_send_sync(mic_epd, &ack, sizeof(ack)) == ERROR) {
        return ERROR;
    }
    
    return SUCCESS;
}

int mic_task_dgemm(mic_task_context_t *this_task)
{
 //   sleep(1);
	return SUCCESS;
}

int mic_task_sgemm(mic_task_context_t *this_task)
{
 //   sleep(0.02);
	return SUCCESS;
}

int mic_server_accept_command()
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
            (mic_command_array[command])();
        }
    }

	return SUCCESS;

}

void *mic_server_progress_task_queue(void *_argv)
{
    mic_task_context_t *this_task;
    int *pid = (int *)_argv;
    printf("my pid %d\n", *pid);

    while(1) {
        if (mic_server_task_queue->runable_task > 0) {
            pthread_mutex_lock(&(mic_server_task_queue->task_queue_lock));
            this_task = mic_server_task_queue->execution;
            if (this_task != NULL) {
                if (this_task->task_status == TASK_READY) {
                    mic_server_task_queue->execution = this_task->next_task;
			        mic_server_task_queue->runable_task --;
                    printf("thread %d grab task id %d, runable task left %d, next task %p\n", *pid, this_task->task_id, mic_server_task_queue->runable_task, this_task->next_task);
                }
            }
            pthread_mutex_unlock(&(mic_server_task_queue->task_queue_lock));

            if (this_task != NULL) {
                (mic_task_array[this_task->task_name])(this_task);
                this_task->task_status = TASK_FINISH;
              //  printf("I finished task %d\n", this_task->task_id);
            }
        }
        
        
    }
}

void *mic_server_finish_task_queue()
{
    mic_task_context_t *this_task;
    int ack = -1;

    while (1) {
        this_task = mic_server_task_queue->head;
        if (this_task != NULL) {
            if (this_task->task_status == TASK_FINISH) {
                pthread_mutex_lock(&(mic_server_task_queue->task_queue_lock));
                mic_server_task_queue->head = this_task->next_task;
                mic_server_task_queue->total_task --;
                pthread_mutex_unlock(&(mic_server_task_queue->task_queue_lock));
                if (this_task->task_data != NULL) {
                    free(this_task->task_data);
                }
                ack = this_task->task_id;
               // if (mic_send_async(mic_epd, &ack, sizeof(ack)) == ERROR) { 
			     //   return ERROR;
		        //}
                free(this_task);
                printf("I send task %d ACK back\n", ack);
            }
        }
    }
}

int mic_server_run()
{
    pthread_t threads[MIC_TASK_NUM_OF_THREADS];
    pthread_t thread_send_ack;
    int rc, i;
    
    for (i = 0; i < MIC_TASK_NUM_OF_THREADS; i++) {
        int *pid = (int *)malloc(sizeof(int));
        *pid = i;
        rc = pthread_create(&threads[i], NULL, mic_server_progress_task_queue, pid);
        if (rc) {
            printf("[Error] can not create thread %d\n", i);
        }
    }

  /*  rc = pthread_create(&thread_send_ack, NULL, mic_server_finish_task_queue, NULL);
    if (rc) {
        printf("[Error] can not create finsh task thread\n");
    }*/

    mic_server_accept_command();
    return SUCCESS;
}

int mic_task_queue_init()
{
    int i;
    for (i = 0; i < COMMAND_FUN_MAX; i++) {
        mic_command_array[i] = NULL;
    }
    mic_command_array[MALLOC_MEM] = mic_command_malloc;
    mic_command_array[CLOSE_MIC] = mic_server_close;
    mic_command_array[MIC_DGEMM] = mic_command_dgemm;
    mic_command_array[MIC_SGEMM] = mic_command_sgemm;
    mic_command_array[CHECK_COMPLETION] = mic_command_check_completion;

    mic_task_array[MIC_DGEMM] = mic_task_dgemm;
    mic_task_array[MIC_SGEMM] = mic_task_sgemm;
    
    mic_server_task_queue = (mic_task_queue_t *)malloc(sizeof(mic_task_queue_t));
	mic_server_task_queue->head = NULL;
    mic_server_task_queue->end = NULL;
    mic_server_task_queue->execution = NULL;
    mic_server_task_queue->total_task = 0;
    mic_server_task_queue->runable_task = 0;
    mic_server_task_queue->finished_task = 0;
	
	pthread_mutex_init(&(mic_server_task_queue->task_queue_lock), NULL);
    pthread_cond_init(&(mic_server_task_queue->task_ready), NULL);

    
    return SUCCESS;
}

int main(int argc, char *argv[])
{
	if (mic_server_init() != SUCCESS) {
		printf("[Error]:server can not start\n");
		exit(1);
	}
    mic_task_queue_init();
	mic_server_run();	

	if (mic_server_close() != SUCCESS) {
		printf("[Error]:server can not close\n");
		exit(1);
	}
	return 0;	
}

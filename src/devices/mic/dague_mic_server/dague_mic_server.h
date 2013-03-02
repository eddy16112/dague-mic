#ifndef DAGUE_MIC_SERVER_H
#define DAGUE_MIC_SERVER_H

#include <scif.h>
#include <pthread.h>

#define SUCCESS			1
#define ERROR			0

#define PAGE_SIZE       0x1000

#define COMMAND_FUN_MAX     100

#define MALLOC_MEM			10
#define CLOSE_MIC			11
#define SYNC_WRITE			12
#define CHECK_COMPLETION    13
#define MIC_DGEMM			20
#define MIC_SGEMM			21

#define TASK_READY			101
#define TASK_EXECUTION		102
#define TASK_FINISH			103

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

typedef struct mic_task_context {
    void *task_data;                         /** data of the task */
    int task_status;                         /** status of the task, TASK_READY, TASK_FINISH */
    struct mic_task_context *next_task;
    pthread_mutex_t task_lock;               /** lock the task when a thread is trying to take it */
	int task_name;                           /** name of task */
    int task_id;                             /** client uses it to identify task */
}mic_task_context_t;

typedef struct mic_task_queue {
    pthread_mutex_t task_queue_lock;         /** lock the queue when a thread is trying to take a task from task queue */
    pthread_cond_t task_ready;               /** cond_t for job ready */
    struct mic_task_context *head;           /** head of task queue, IN */ 
    struct mic_task_context *end;            /** end of task queue, OUT */
    struct mic_task_context *execution;      /** this task will be executed next */
    int total_task;                          /** number of total tasks in queue */
    int runable_task;                        /** number of tasks could be executed */
    int finished_task;
}mic_task_queue_t;

typedef struct mic_gemm_param {
	void* gemm_A;
	void* gemm_B;
	void* gemm_C;
	int m;
	int n;
	int k;
    int task_id;                              
}mic_gemm_param_t;

typedef int micEvent_t;

typedef int (*mic_command_function_t)();

typedef int (*mic_task_function_t)(mic_task_context_t *this_task);

/** send and receive command to/from client */
int mic_send_sync(scif_epd_t mepd, void *msg, int len);
int mic_send_async(scif_epd_t mepd, void *msg, int len);
int mic_recv_sync(scif_epd_t mepd, void *msg, int len);

/** malloc mem in mic */
int mic_malloc_server(void **addr, size_t nbyte, off_t *offset);

/** accept command from client */
int mic_command_array_init();
int mic_server_accept_command();
int mic_server_task_enqueue(mic_task_context_t * this_task);
int mic_command_malloc();
int mic_command_dgemm();
int mic_command_sgemm();
int mic_command_check_completion();

/** process task in queue */
void *mic_server_progress_task_queue(void *_argv);
int mic_task_dgemm(mic_task_context_t * this_task);
int mic_task_sgemm(mic_task_context_t * this_task);

/** finish task in queue, send comfirmation back to client */
void *mic_server_finish_task_queue();

/** init and close server */
int mic_server_init();
int mic_server_close();
int mic_server_run();

#endif

#include "nccl.h"
#include "time.h"

void sleep_us(unsigned long microseconds)
{
    struct timespec ts;
    ts.tv_sec = microseconds / 1000000ul;            // whole seconds
    ts.tv_nsec = (microseconds % 1000000ul) * 1000;  // remainder, in nanoseconds
    nanosleep(&ts, NULL);
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    sleep_us(1000);
    return ncclSuccess;
}

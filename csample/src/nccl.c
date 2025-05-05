#include "nccl.h"
#include "stdint.h"
#include "stdio.h"
void sleep_kernel(int64_t num_cycles, cudaStream_t stream);


ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    sleep_kernel(100000, stream);
    return ncclSuccess;
}

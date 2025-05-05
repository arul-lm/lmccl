#include "nccl.h"

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    clock_t start = clock();
    clock_t now;
    for (;;) {
        now = clock();
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= 10000) {
            break;
        }
    }
// Stored "now" in global memory here to prevent the
// compiler from optimizing away the entire loop.
    *global_now = now;
    return ncclSystemError;
}

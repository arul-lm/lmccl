// Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef NCCL_H_
#define NCCL_H_
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;
/* Error type */
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;
typedef enum {
  ncclSum = 0,
  ncclProd = 1,
  ncclMax = 2,
  ncclMin = 3,
  ncclAvg = 4,
  /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
   * serves as the least possible value for dynamic ncclRedOp_t's
   * as constructed by ncclRedOpCreate*** functions. */
  ncclNumOps = 5,
  /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
   * It is defined to be the largest signed value (since compilers
   * are permitted to use signed enums) that won't grow
   * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
   * maintain ABI compatibility. */
  ncclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(ncclRedOp_dummy_t))
} ncclRedOp_t;

/* Data types */
typedef enum {
  ncclInt8 = 0,
  ncclChar = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclInt = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclHalf = 6,
  ncclFloat32 = 7,
  ncclFloat = 7,
  ncclFloat64 = 8,
  ncclDouble = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__)
  ncclBfloat16 = 9,
  ncclFp8E4M3 = 10,
  ncclFp8E5M2 = 11,
  ncclNumTypes = 12
#elif defined(__CUDA_BF16_TYPES_EXIST__)
  ncclBfloat16 = 9,
  ncclNumTypes = 10
#else
  ncclNumTypes = 9
#endif
} ncclDataType_t;

typedef int cudaStream_t;
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // end include guard

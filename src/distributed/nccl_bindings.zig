const std = @import("std");

pub const ncclResult_t = enum(c_int) {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8,
};

pub const ncclDataType_t = enum(c_int) {
    ncclInt8 = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclFloat32 = 7,
    ncclFloat64 = 8,
    ncclBfloat16 = 9,
    ncclNumTypes = 10,
};

pub const ncclChar = ncclDataType_t.ncclInt8;
pub const ncclInt = ncclDataType_t.ncclInt32;
pub const ncclHalf = ncclDataType_t.ncclFloat16;
pub const ncclFloat = ncclDataType_t.ncclFloat32;
pub const ncclDouble = ncclDataType_t.ncclFloat64;

pub const ncclRedOp_t = enum(c_int) {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5,
};

pub const ncclComm = opaque {};
pub const ncclUniqueId = extern struct {
    internal: [128]u8,
};

pub extern fn ncclGetUniqueId(uniqueId: *ncclUniqueId) ncclResult_t;
pub extern fn ncclCommInitRank(comm: **ncclComm, nranks: c_int, commId: ncclUniqueId, rank: c_int) ncclResult_t;
pub extern fn ncclCommDestroy(comm: *ncclComm) ncclResult_t;
pub extern fn ncclCommFinalize(comm: *ncclComm) ncclResult_t;
pub extern fn ncclCommCount(comm: *const ncclComm, count: *c_int) ncclResult_t;
pub extern fn ncclCommCuDevice(comm: *const ncclComm, device: *c_int) ncclResult_t;
pub extern fn ncclCommUserRank(comm: *const ncclComm, rank: *c_int) ncclResult_t;

pub extern fn ncclAllReduce(
    sendbuff: ?*const anyopaque,
    recvbuff: ?*anyopaque,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: *ncclComm,
    stream: ?*anyopaque,
) ncclResult_t;

pub extern fn ncclBroadcast(
    sendbuff: ?*const anyopaque,
    recvbuff: ?*anyopaque,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: *ncclComm,
    stream: ?*anyopaque,
) ncclResult_t;

pub extern fn ncclReduce(
    sendbuff: ?*const anyopaque,
    recvbuff: ?*anyopaque,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: *ncclComm,
    stream: ?*anyopaque,
) ncclResult_t;

pub extern fn ncclAllGather(
    sendbuff: ?*const anyopaque,
    recvbuff: ?*anyopaque,
    sendcount: usize,
    datatype: ncclDataType_t,
    comm: *ncclComm,
    stream: ?*anyopaque,
) ncclResult_t;

pub extern fn ncclReduceScatter(
    sendbuff: ?*const anyopaque,
    recvbuff: ?*anyopaque,
    recvcount: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: *ncclComm,
    stream: ?*anyopaque,
) ncclResult_t;

pub extern fn ncclGetErrorString(result: ncclResult_t) [*:0]const u8;

pub const CudaError = enum(c_int) {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    _,
};

pub extern fn cudaGetDeviceCount(count: *c_int) CudaError;
pub extern fn cudaSetDevice(device: c_int) CudaError;
pub extern fn cudaGetDevice(device: *c_int) CudaError;
pub extern fn cudaMalloc(devPtr: *?*anyopaque, size: usize) CudaError;
pub extern fn cudaFree(devPtr: ?*anyopaque) CudaError;
pub extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int) CudaError;
pub extern fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) CudaError;
pub extern fn cudaDeviceSynchronize() CudaError;
pub extern fn cudaStreamCreate(pStream: **anyopaque) CudaError;
pub extern fn cudaStreamDestroy(stream: *anyopaque) CudaError;
pub extern fn cudaStreamSynchronize(stream: *anyopaque) CudaError;
pub extern fn cudaGetErrorString(err: CudaError) [*:0]const u8;
pub extern fn cudaGetLastError() CudaError;

pub const cudaMemcpyKind = struct {
    pub const cudaMemcpyHostToDevice: c_int = 1;
    pub const cudaMemcpyDeviceToHost: c_int = 2;
    pub const cudaMemcpyDeviceToDevice: c_int = 3;
};


const _build_gpu_enabled: bool = blk: {
    const opts = @import("build_options");
    if (@hasDecl(opts, "gpu_acceleration")) break :blk opts.gpu_acceleration;
    break :blk false;
};

pub const cudaError_t = c_uint;
pub const cudaSuccess: cudaError_t = 0;
pub const cudaErrorInvalidValue: cudaError_t = 1;
pub const cudaErrorMemoryAllocation: cudaError_t = 2;
pub const cudaErrorInitializationError: cudaError_t = 3;
pub const cudaErrorLaunchFailure: cudaError_t = 4;
pub const cudaErrorLaunchTimeout: cudaError_t = 6;
pub const cudaErrorLaunchOutOfResources: cudaError_t = 7;
pub const cudaErrorInvalidDeviceFunction: cudaError_t = 8;
pub const cudaErrorInvalidConfiguration: cudaError_t = 9;
pub const cudaErrorInvalidDevice: cudaError_t = 10;
pub const cudaErrorInvalidMemcpyDirection: cudaError_t = 21;

pub const cudaHostAllocDefault: c_uint = 0;
pub const cudaHostAllocPortable: c_uint = 1;
pub const cudaHostAllocMapped: c_uint = 2;
pub const cudaHostAllocWriteCombined: c_uint = 4;

pub const cudaMemcpyHostToHost: c_uint = 0;
pub const cudaMemcpyHostToDevice: c_uint = 1;
pub const cudaMemcpyDeviceToHost: c_uint = 2;
pub const cudaMemcpyDeviceToDevice: c_uint = 3;
pub const cudaMemcpyDefault: c_uint = 4;

pub const cudaStream_t = ?*anyopaque;

pub const CudaError = error{
    InvalidValue,
    MemoryAllocation,
    InitializationError,
    LaunchFailure,
    LaunchTimeout,
    LaunchOutOfResources,
    InvalidDeviceFunction,
    InvalidConfiguration,
    InvalidDevice,
    InvalidMemcpyDirection,
    HostAllocFailed,
    Unknown,
};

const RealApi = struct {
    pub extern "c" fn cudaHostAlloc(ptr: *?*anyopaque, size: usize, flags: c_uint) cudaError_t;
    pub extern "c" fn cudaFreeHost(ptr: ?*anyopaque) cudaError_t;
    pub extern "c" fn cudaMalloc(devPtr: *?*anyopaque, size: usize) cudaError_t;
    pub extern "c" fn cudaFree(devPtr: ?*anyopaque) cudaError_t;
    pub extern "c" fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint) cudaError_t;
    pub extern "c" fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint, stream: cudaStream_t) cudaError_t;
    pub extern "c" fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) cudaError_t;
    pub extern "c" fn cudaDeviceSynchronize() cudaError_t;
    pub extern "c" fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t;
    pub extern "c" fn cudaGetLastError() cudaError_t;
    pub extern "c" fn cudaPeekAtLastError() cudaError_t;
    pub extern "c" fn cudaGetErrorString(err: cudaError_t) [*:0]const u8;
    pub extern "c" fn cudaGetErrorName(err: cudaError_t) [*:0]const u8;
    pub extern "c" fn cudaStreamCreate(pStream: *cudaStream_t) cudaError_t;
    pub extern "c" fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t;
    pub extern "c" fn cudaGetDeviceCount(count: *c_int) cudaError_t;
    pub extern "c" fn cudaSetDevice(device: c_int) cudaError_t;
    pub extern "c" fn cudaGetDevice(device: *c_int) cudaError_t;
};

const StubApi = struct {
    pub fn cudaHostAlloc(ptr: *?*anyopaque, size: usize, flags: c_uint) cudaError_t {
        _ = ptr;
        _ = size;
        _ = flags;
        return cudaErrorInitializationError;
    }
    pub fn cudaFreeHost(ptr: ?*anyopaque) cudaError_t {
        _ = ptr;
        return cudaSuccess;
    }
    pub fn cudaMalloc(devPtr: *?*anyopaque, size: usize) cudaError_t {
        _ = devPtr;
        _ = size;
        return cudaErrorInitializationError;
    }
    pub fn cudaFree(devPtr: ?*anyopaque) cudaError_t {
        _ = devPtr;
        return cudaSuccess;
    }
    pub fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint) cudaError_t {
        _ = dst;
        _ = src;
        _ = count;
        _ = kind;
        return cudaErrorInitializationError;
    }
    pub fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint, stream: cudaStream_t) cudaError_t {
        _ = dst;
        _ = src;
        _ = count;
        _ = kind;
        _ = stream;
        return cudaErrorInitializationError;
    }
    pub fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) cudaError_t {
        _ = devPtr;
        _ = value;
        _ = count;
        return cudaErrorInitializationError;
    }
    pub fn cudaDeviceSynchronize() cudaError_t {
        return cudaSuccess;
    }
    pub fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t {
        _ = stream;
        return cudaSuccess;
    }
    pub fn cudaGetLastError() cudaError_t {
        return cudaSuccess;
    }
    pub fn cudaPeekAtLastError() cudaError_t {
        return cudaSuccess;
    }
    pub fn cudaGetErrorString(err: cudaError_t) [*:0]const u8 {
        _ = err;
        return "cuda stub (gpu disabled)";
    }
    pub fn cudaGetErrorName(err: cudaError_t) [*:0]const u8 {
        _ = err;
        return "cudaStub";
    }
    pub fn cudaStreamCreate(pStream: *cudaStream_t) cudaError_t {
        _ = pStream;
        return cudaErrorInitializationError;
    }
    pub fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t {
        _ = stream;
        return cudaSuccess;
    }
    pub fn cudaGetDeviceCount(count: *c_int) cudaError_t {
        count.* = 0;
        return cudaSuccess;
    }
    pub fn cudaSetDevice(device: c_int) cudaError_t {
        _ = device;
        return cudaErrorInitializationError;
    }
    pub fn cudaGetDevice(device: *c_int) cudaError_t {
        device.* = -1;
        return cudaErrorInitializationError;
    }
};

pub const api = if (_build_gpu_enabled) RealApi else StubApi;

pub const cudaHostAlloc = api.cudaHostAlloc;
pub const cudaFreeHost = api.cudaFreeHost;
pub const cudaMalloc = api.cudaMalloc;
pub const cudaFree = api.cudaFree;
pub const cudaMemcpy = api.cudaMemcpy;
pub const cudaMemcpyAsync = api.cudaMemcpyAsync;
pub const cudaMemset = api.cudaMemset;
pub const cudaDeviceSynchronize = api.cudaDeviceSynchronize;
pub const cudaStreamSynchronize = api.cudaStreamSynchronize;
pub const cudaGetLastError = api.cudaGetLastError;
pub const cudaPeekAtLastError = api.cudaPeekAtLastError;
pub const cudaGetErrorString = api.cudaGetErrorString;
pub const cudaGetErrorName = api.cudaGetErrorName;
pub const cudaStreamCreate = api.cudaStreamCreate;
pub const cudaStreamDestroy = api.cudaStreamDestroy;
pub const cudaGetDeviceCount = api.cudaGetDeviceCount;
pub const cudaSetDevice = api.cudaSetDevice;
pub const cudaGetDevice = api.cudaGetDevice;

const std = @import("std");
const InferenceServer = @import("api/inference_server.zig").InferenceServer;
const ServerConfig = @import("api/inference_server.zig").ServerConfig;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var config = ServerConfig{
        .port = 8080,
        .host = "0.0.0.0",
        .max_connections = 100,
        .request_timeout_ms = 30000,
        .batch_size = 32,
        .model_path = null,
        .rate_limit_per_minute = 60,
        .max_request_size_bytes = 1024 * 1024,
        .require_api_key = false,
    };

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            i += 1;
            config.port = std.fmt.parseInt(u16, args[i], 10) catch 8080;
        } else if (std.mem.eql(u8, arg, "--host") and i + 1 < args.len) {
            i += 1;
            config.host = args[i];
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            i += 1;
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--require-api-key")) {
            config.require_api_key = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("JAIDE Inference Server\n\n", .{});
            std.debug.print("Options:\n", .{});
            std.debug.print("  --port <port>       Port to listen on (default: 8080)\n", .{});
            std.debug.print("  --host <host>       Host to bind to (default: 0.0.0.0)\n", .{});
            std.debug.print("  --model <path>      Path to model file\n", .{});
            std.debug.print("  --dataset <path>    Path to dataset file\n", .{});
            std.debug.print("  --require-api-key   Require API key for requests\n", .{});
            std.debug.print("  --help              Show this help message\n", .{});
            return;
        }
    }

    if (std.posix.getenv("JAIDE_MODEL_PATH")) |model_path| {
        if (config.model_path == null) {
            config.model_path = model_path;
        }
    }

    var server = InferenceServer.init(allocator, config) catch |err| {
        std.debug.print("Failed to initialize server: {}\n", .{err});
        return err;
    };
    defer server.deinit();

    std.debug.print("Starting JAIDE Inference Server on {s}:{d}\n", .{ config.host, config.port });

    if (config.model_path) |path| {
        server.loadModel(path) catch |err| {
            std.debug.print("Failed to load model from {s}: {}\n", .{ path, err });
        };
    }

    server.start() catch |err| {
        std.debug.print("Server error: {}\n", .{err});
        return err;
    };
}


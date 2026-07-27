const std = @import("std");
const http = std.http;

pub const ModalGPUClient = struct {
    allocator: std.mem.Allocator,
    api_token: []const u8,
    http_client: http.Client,
    gpu_count: usize,
    gpu_preferences: [2][]const u8,

    pub fn init(allocator: std.mem.Allocator, api_token: []const u8) !ModalGPUClient {
        return .{
            .allocator = allocator,
            .api_token = try allocator.dupe(u8, api_token),
            .http_client = http.Client{ .allocator = allocator },
            .gpu_count = 8,
            .gpu_preferences = .{ "B300", "B200" },
        };
    }

    pub fn deinit(self: *ModalGPUClient) void {
        self.allocator.free(self.api_token);
        self.http_client.deinit();
    }

    pub fn deployTrainingJob(self: *ModalGPUClient, model_path: []const u8, dataset_path: []const u8) ![]const u8 {
        const payload = try std.json.stringifyAlloc(self.allocator, .{
            .gpu = self.gpu_preferences,
            .gpu_count = self.gpu_count,
            .image = "jaide-v40-training",
            .model_path = model_path,
            .dataset_path = dataset_path,
            .batch_size = 32,
            .epochs = 10,
        }, .{});
        defer self.allocator.free(payload);

        return try self.sendRequest(.POST, "https://api.modal.com/v1/functions/deploy", payload);
    }

    pub fn getJobStatus(self: *ModalGPUClient, job_id: []const u8) ![]const u8 {
        const uri_str = try std.fmt.allocPrint(self.allocator, "https://api.modal.com/v1/functions/{s}/status", .{job_id});
        defer self.allocator.free(uri_str);

        return try self.sendRequest(.GET, uri_str, null);
    }

    fn sendRequest(self: *ModalGPUClient, method: http.Method, url: []const u8, body: ?[]const u8) ![]const u8 {
        const authorization_value = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_token});
        defer self.allocator.free(authorization_value);

        var response_storage = std.ArrayList(u8).init(self.allocator);
        errdefer response_storage.deinit();

        const fetch_options = if (body) |b|
            http.Client.FetchOptions{
                .method = method,
                .location = .{ .url = url },
                .headers = .{
                    .authorization = .{ .override = authorization_value },
                    .content_type = .{ .override = "application/json" },
                },
                .payload = b,
                .response_storage = .{ .dynamic = &response_storage },
                .max_append_size = 1024 * 1024,
            }
        else
            http.Client.FetchOptions{
                .method = method,
                .location = .{ .url = url },
                .headers = .{
                    .authorization = .{ .override = authorization_value },
                },
                .response_storage = .{ .dynamic = &response_storage },
                .max_append_size = 1024 * 1024,
            };

        const result = try self.http_client.fetch(fetch_options);
        _ = result;

        return response_storage.toOwnedSlice();
    }
};

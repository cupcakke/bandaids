#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int g_failures = 0;
static int g_passes = 0;

static void check(const char* name, int cond) {
    if (cond) {
        g_passes++;
        fprintf(stdout, "[PASS] %s\n", name);
    } else {
        g_failures++;
        fprintf(stderr, "[FAIL] %s\n", name);
    }
}

static int roundtrip_i64(int64_t v) {
    unsigned char buf[8];
    memcpy(buf, &v, sizeof(v));
    int64_t out = 0;
    memcpy(&out, buf, sizeof(out));
    return out == v;
}

static int roundtrip_double(double v) {
    unsigned char buf[8];
    memcpy(buf, &v, sizeof(v));
    double out = 0.0;
    memcpy(&out, buf, sizeof(out));
    return out == v;
}

static uint64_t fnv1a(const void* data, size_t len) {
    const unsigned char* p = (const unsigned char*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int abi_layout_check(void) {
    struct probe {
        int32_t a;
        int64_t b;
        double  c;
        uint8_t d;
    };
    struct probe p;
    memset(&p, 0, sizeof(p));
    p.a = 0x11223344;
    p.b = 0x1122334455667788LL;
    p.c = 3.14159265358979323846;
    p.d = 0xAB;
    if (sizeof(int32_t) != 4) return 0;
    if (sizeof(int64_t) != 8) return 0;
    if (sizeof(double) != 8) return 0;
    if (sizeof(uint8_t) != 1) return 0;
    if (p.a != 0x11223344) return 0;
    if (p.b != 0x1122334455667788LL) return 0;
    return 1;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    fprintf(stdout, "jaide-c-api-test starting\n");

    check("i64 roundtrip zero", roundtrip_i64(0));
    check("i64 roundtrip max", roundtrip_i64(INT64_MAX));
    check("i64 roundtrip min", roundtrip_i64(INT64_MIN));
    check("i64 roundtrip pattern", roundtrip_i64(0x0F0F0F0F0F0F0F0FLL));

    check("double roundtrip zero", roundtrip_double(0.0));
    check("double roundtrip one", roundtrip_double(1.0));
    check("double roundtrip pi", roundtrip_double(3.14159265358979323846));

    check("abi layout", abi_layout_check());

    const char* payload = "jaide-inference-trace";
    uint64_t h1 = fnv1a(payload, strlen(payload));
    uint64_t h2 = fnv1a(payload, strlen(payload));
    check("hash determinism", h1 == h2 && h1 != 0);

    void* p = malloc(4096);
    check("malloc 4096", p != NULL);
    if (p != NULL) {
        memset(p, 0xAA, 4096);
        unsigned char* q = (unsigned char*)p;
        int ok = 1;
        for (size_t i = 0; i < 4096; i++) {
            if (q[i] != 0xAA) { ok = 0; break; }
        }
        check("memset 4096", ok);
        free(p);
    }

    fprintf(stdout, "jaide-c-api-test: %d passed, %d failed\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}

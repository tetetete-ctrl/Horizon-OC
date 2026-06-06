/*
 * Copyright © 2011 Siarhei Siamashka <siarhei.siamashka@gmail.com>
 * Copyright (c) 20xx KazushiMe
 * Copyright (c) 2025 Souldbminer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/time.h>

#define SIZE (32 * 1024 * 1024)
#ifndef MAXREPEATS
    #define MAXREPEATS 10
#endif
#ifndef LATBENCH_COUNT
    #define LATBENCH_COUNT 10000000
#endif
#define ALIGN_PADDING 0x100000
#define CACHE_LINE_SIZE 128

#include <switch.h>

#include "gpu_bw.h"

#define YEL "\033[1;92m"
#define GRN "\033[1;92m"
#define RST "\033[0m"

PadState pad;

// Threads

struct f_data {
    void (*func)(int64_t *, int64_t *, int);
    int64_t *arg1;
    int64_t *arg2;
    int arg3;
};

pthread_cond_t p_ready, p_start;
pthread_mutex_t p_lock;
pthread_t *p_worker = NULL;
struct f_data *worker_data = NULL;
int p_worker_not_ready, p_workers_ready;

void *thread_func(void *data) {
    struct f_data *d = data;
    pthread_mutex_lock(&p_lock);
    p_worker_not_ready--;
    if (!p_worker_not_ready)
        pthread_cond_signal(&p_ready);
    while (p_workers_ready != 1)
        pthread_cond_wait(&p_start, &p_lock);
    pthread_mutex_unlock(&p_lock);
    (d->func)(d->arg1, d->arg2, d->arg3);
    pthread_exit(NULL);
}

static void parallel_run(void) {
    pthread_mutex_lock(&p_lock);
    p_workers_ready = 1;
    pthread_mutex_unlock(&p_lock);
    pthread_cond_broadcast(&p_start);
}

static void parallel_init(int threads) {
    pthread_attr_t attr;
    pthread_cond_init(&p_ready, NULL);
    pthread_cond_init(&p_start, NULL);
    pthread_mutex_init(&p_lock, NULL);
    p_worker_not_ready = threads;
    p_workers_ready = 0;
    pthread_attr_init(&attr);
    if (!p_worker || !worker_data) {
        p_worker = malloc(threads * sizeof(pthread_t));
        worker_data = malloc(threads * sizeof(struct f_data));
    }
    for (int i = 0; i < threads; i++)
        pthread_create(p_worker + i, &attr, thread_func, worker_data + i);
    pthread_mutex_lock(&p_lock);
    while (p_worker_not_ready != 0)
        pthread_cond_wait(&p_ready, &p_lock);
    pthread_mutex_unlock(&p_lock);
}

// memops

void aligned_block_copy(int64_t *__restrict dst_, int64_t *__restrict src, int size) {
    volatile int64_t *dst = dst_;
    int64_t t1, t2, t3, t4;
    while ((size -= 64) >= 0) {
        t1 = *src++;
        t2 = *src++;
        t3 = *src++;
        t4 = *src++;
        *dst++ = t1;
        *dst++ = t2;
        *dst++ = t3;
        *dst++ = t4;
        t1 = *src++;
        t2 = *src++;
        t3 = *src++;
        t4 = *src++;
        *dst++ = t1;
        *dst++ = t2;
        *dst++ = t3;
        *dst++ = t4;
    }
}

void aligned_block_fetch(int64_t *__restrict dst, int64_t *__restrict src_, int size) {
    volatile int64_t *src = src_;
    (void)dst;
    while ((size -= 64) >= 0) {
        *src++;
        *src++;
        *src++;
        *src++;
        *src++;
        *src++;
        *src++;
        *src++;
    }
}

void aligned_block_fill(int64_t *__restrict dst_, int64_t *__restrict src, int size) {
    volatile int64_t *dst = dst_;
    int64_t data = *src;
    while ((size -= 64) >= 0) {
        *dst++ = data;
        *dst++ = data;
        *dst++ = data;
        *dst++ = data;
        *dst++ = data;
        *dst++ = data;
        *dst++ = data;
        *dst++ = data;
    }
}

double gettime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)((int64_t)tv.tv_sec * 1000000 + tv.tv_usec) / 1000000.;
}

static double bandwidth_bench_helper(int threads, int64_t *dstbuf, int64_t *srcbuf, int size, void (*f)(int64_t *, int64_t *, int)) {
    int i, loopcount, innerloopcount, n;
    double t, t1, t2, speed, maxspeed, s, s0, s1, s2;

    s = s0 = s1 = s2 = 0.;
    maxspeed = 0.;
    for (n = 0; n < MAXREPEATS; n++) {
        loopcount = 0;
        innerloopcount = 1;
        t = 0.;
        do {
            loopcount += innerloopcount;
            for (i = 0; i < innerloopcount; i++) {
                parallel_init(threads);
                for (int pt = 0; pt < threads; pt++) {
                    (worker_data + pt)->func = f;
                    (worker_data + pt)->arg1 = dstbuf + size * pt / sizeof(int64_t);
                    (worker_data + pt)->arg2 = srcbuf + size * pt / sizeof(int64_t);
                    (worker_data + pt)->arg3 = size;
                }
                t1 = gettime();
                parallel_run();
                for (int pt = 0; pt < threads; pt++)
                    pthread_join(p_worker[pt], NULL);
                t2 = gettime();
                t += t2 - t1;
            }
            innerloopcount *= 2;
        } while (t < 0.5);

        speed = (double)size * threads * loopcount / t / 1000000.;
        s0 += 1.;
        s1 += speed;
        s2 += speed * speed;
        if (speed > maxspeed)
            maxspeed = speed;
        if (s0 > 2.) {
            s = sqrt((s0 * s2 - s1 * s1) / (s0 * (s0 - 1)));
            if (s < maxspeed / 1000.)
                break;
        }
    }
    return maxspeed;
}

static char *align_up(char *ptr, int align) {
    return (char *)(((uintptr_t)ptr + align - 1) & ~(uintptr_t)(align - 1));
}

void *alloc_nonaliased_buffers(void **buf1_, int size1, void **buf2_, int size2, void **buf3_, int size3) {
    char **buf1 = (char **)buf1_, **buf2 = (char **)buf2_, **buf3 = (char **)buf3_;
    int mask = (ALIGN_PADDING - 1) & ~(CACHE_LINE_SIZE - 1);
    char *buf = malloc(size1 + size2 + size3 + 9 * ALIGN_PADDING);
    char *ptr = buf;
    memset(buf, 0xCC, size1 + size2 + size3 + 9 * ALIGN_PADDING);
    ptr = align_up(ptr, ALIGN_PADDING);
    if (buf1) {
        *buf1 = ptr + (0xAAAAAAAA & mask);
        ptr = align_up(*buf1 + size1, ALIGN_PADDING);
    }
    if (buf2) {
        *buf2 = ptr + (0x55555555 & mask);
        ptr = align_up(*buf2 + size2, ALIGN_PADDING);
    }
    if (buf3) {
        *buf3 = ptr + (0xCCCCCCCC & mask);
    }
    return buf;
}

#pragma GCC diagnostic push
static void __attribute__((noinline)) random_read_test(char *buf, int count, int nbits) {
    uint32_t seed = 0;
    uintptr_t mask = (1 << nbits) - 1;
    uint32_t v;
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    static volatile uint32_t dummy;
#define RMA()                         \
    seed = seed * 1103515245 + 12345; \
    v = (seed >> 16) & 0xFF;          \
    seed = seed * 1103515245 + 12345; \
    v |= (seed >> 8) & 0xFF00;        \
    seed = seed * 1103515245 + 12345; \
    v |= seed & 0x7FFF0000;           \
    seed |= buf[v & mask];
    while (count >= 16) {
        RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() RMA() count -= 16;
    }
    dummy = seed;
#undef RMA
}
#pragma GCC diagnostic pop

static double latency_measure(char *buf, int nbits, int count) {
    double t_noaccess = 0, t_before, t_after, t, xs1 = 0, xs2 = 0, min_t = 0;
    double xs;
    int n;

    for (n = 1; n <= MAXREPEATS; n++) {
        t_before = gettime();
        random_read_test(buf, count, 1);
        t_after = gettime();
        if (n == 1 || t_after - t_before < t_noaccess)
            t_noaccess = t_after - t_before;
    }

    for (n = 1; n <= MAXREPEATS; n++) {
        t_before = gettime();
        random_read_test(buf, count, nbits);
        t_after = gettime();
        t = t_after - t_before - t_noaccess;
        if (t < 0)
            t = 0;
        xs1 += t;
        xs2 += t * t;
        if (n == 1 || t < min_t)
            min_t = t;
        if (n > 2) {
            xs = sqrt((xs2 * n - xs1 * xs1) / (n * (n - 1)));
            if (xs < min_t / 1000.)
                break;
        }
    }
    return min_t * 1000000000.0 / count;
}

static void latency_bench(double *l2_out, double *ram_out) {
    char *buf_alloc = malloc(0x2001000);
    char *buf = (char *)(((uintptr_t)buf_alloc + 4095) & ~(uintptr_t)4095);
    memset(buf, 0, 0x2000000);

    /* nbits=20 (1MB) for L2, nbits=25 (32MB) for RAM — from binary assembly */
    *l2_out = latency_measure(buf, 20, LATBENCH_COUNT);
    *ram_out = latency_measure(buf, 25, LATBENCH_COUNT);

    free(buf_alloc);
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    consoleInit(NULL);
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    padInitializeDefault(&pad);

    const int threads = 3;
    const int size = SIZE;
    bool is_4gb = (appletGetAppletType() == AppletType_Application);

loop:
    printf("Membench-NX\n\n");
    printf("Press A to run all benchmarks.\n");
    printf("Press any other key to exit.\n\n");
    consoleUpdate(NULL);

    while (appletMainLoop()) {
        padUpdate(&pad);
        u64 kDown = padGetButtonsDown(&pad);
        if (kDown & HidNpadButton_A)
            break;
        if (kDown) {
            consoleExit(NULL);
            return 0;
        }
        consoleUpdate(NULL);
    }

    appletSetAutoSleepDisabled(true);
    consoleClear();

    uint32_t cpu_hz = 0, gpu_hz = 0, mem_hz = 0;
    if (R_SUCCEEDED(clkrstInitialize())) {
        ClkrstSession s;
        clkrstOpenSession(&s, PcvModuleId_CpuBus, 3);
        clkrstGetClockRate(&s, &cpu_hz);
        clkrstCloseSession(&s);
        clkrstOpenSession(&s, PcvModuleId_GPU, 3);
        clkrstGetClockRate(&s, &gpu_hz);
        clkrstCloseSession(&s);
        clkrstOpenSession(&s, PcvModuleId_EMC, 3);
        clkrstGetClockRate(&s, &mem_hz);
        clkrstCloseSession(&s);
        clkrstExit();
    }

    printf("Membench-NX\n\n");
    printf("----------------------------\n");
    printf("CPU:      %.1f MHz\n", cpu_hz / 1000000.0);
    printf("GPU:      %.1f MHz\n", gpu_hz / 1000000.0);
    printf("MEM:      %.1f MHz\n", mem_hz / 1000000.0);
    printf("Mode:     " GRN "%s" RST "\n", is_4gb ? "Application" : "Applet");
    printf("Threads:  %d (max: %d)\n", threads, threads);
    printf("----------------------------\n");
    consoleUpdate(NULL);

    printf("\nStarted Bandwidth benchmark via GPU...\n");
    consoleUpdate(NULL);

    double gpu_copy = 0, gpu_read = 0, gpu_write = 0;
    gpu_bw_run(is_4gb, &gpu_copy, &gpu_read, &gpu_write);

    printf(" " YEL "%-40s" RST " : %8.1f MB/s\n", "GPU Copy", gpu_copy);
    printf(" " YEL "%-40s" RST " : %8.1f MB/s\n", "GPU Read", gpu_read);
    printf(" " YEL "%-40s" RST " : %8.1f MB/s\n", "GPU Write", gpu_write);
    consoleUpdate(NULL);

    int64_t *srcbuf, *dstbuf;
    void *poolbuf = alloc_nonaliased_buffers((void **)&srcbuf, size * threads, (void **)&dstbuf, size * threads, NULL, 0);

    printf("\nStarted Bandwidth benchmark with %d threads...\n", threads);
    consoleUpdate(NULL);

    double cpu_copy = bandwidth_bench_helper(threads, dstbuf, srcbuf, size, aligned_block_copy);
    double cpu_read = bandwidth_bench_helper(threads, dstbuf, srcbuf, size, aligned_block_fetch);
    double cpu_write = bandwidth_bench_helper(threads, dstbuf, srcbuf, size, aligned_block_fill);

    printf(" " YEL "%-40s" RST " : %8.1f MB/s\n", "CPU Copy", cpu_copy);
    printf(" " YEL "%-40s" RST " : %8.1f MB/s\n", "CPU Read", cpu_read);
    printf(" " YEL "%-40s" RST " : %8.1f MB/s\n", "CPU Write", cpu_write);
    consoleUpdate(NULL);

    free(poolbuf);

    printf("\nStarted Latency benchmark with 1 thread...\n");
    consoleUpdate(NULL);

    double l2_ns = 0, ram_ns = 0;
    latency_bench(&l2_ns, &ram_ns);

    printf(" " YEL "%-40s" RST " : %8.1f ns\n", "L2", l2_ns);
    printf(" " YEL "%-40s" RST " : %8.1f ns\n", "Full RAM", ram_ns);
    consoleUpdate(NULL);

    appletSetAutoSleepDisabled(false);

    printf("\nPress A to continue, any other key to exit.\n");
    consoleUpdate(NULL);

    while (appletMainLoop()) {
        padUpdate(&pad);
        u64 kDown = padGetButtonsDown(&pad);
        if (kDown & HidNpadButton_A) {
            consoleClear();
            goto loop;
        }
        if (kDown)
            break;
        consoleUpdate(NULL);
    }

    consoleExit(NULL);
    return 0;
}

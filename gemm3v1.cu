// #include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <random>
#include <memory>
#include <sys/types.h>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

#define kernelCheck(op) __kernel_check(__FILE__,__LINE__)
static void __kernel_check(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        // exit(1);
    }
}


#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#define CP_ASYNC_CA(dst,src,Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(dst), "l"(src),"n"(Bytes));
// Asynchronous replication from global memory to shared memory
#define CP_ASYNC_CG(dst,src,Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(dst), "l"(src),"n"(Bytes));
// Asynchronous replication from shared memory to global memory
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

#define THREADS_PER_BLOCK 256
#define WARP_PER_BLOCK 8
#define div_ceil(a,b) (((a)+(b)-1)/(b))
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define THREAD_COPY_BYTES 16

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_COL_TILES 16 //BLOCK_M/MMA_M
#define BLOCK_ROW_TILES 16 //BLOCK_N/MMA_N
#define Stage 3
#define Block_M 256
#define Block_N 128
#define B_ 8
#define M_ 8
#define S_ 8

// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:02:28 on Tue, Feb 28, 2023
//
// Description: mma async stage4 hgemm


#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 16  // BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / MMA_M

#define WARP_ROW_TILES 8  // WARP_COLS / MMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / MMA_K

#define THREAD_COPY_BYTES 16

#define CHUNK_LINE_BYTES 64          // CHUNK_K * MMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

#define SMEM_BANK_ROWS 2  // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4

#define K_STAGE 3

__global__ void mmaAsyncStage3Kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                     size_t M, size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;
    constexpr size_t smem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    half *smem_warp_tile_row_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS;
    const half *smem_warp_stream_ptr = &smem[0][0] + warp_id * MMA_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t smem_store_idx = 0;
    size_t smem_load_idx = 0;

    size_t smem_store_off = 0;
    size_t smem_load_off = 0;

    size_t A_smem_idx = 0;
    int4 *A_lane_ptr = nullptr;

    size_t B_smem_idx = 0;
    int4 *B_lane_ptr = nullptr;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);// global address 使用 int4*

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K); 
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();

    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(1);

    __syncthreads();

    uint32_t RA[2][WARP_COL_TILES][4];
    uint32_t RB[2][WARP_ROW_TILES][2];

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
            &smem[A_smem_idx + lane_id % 16][((lane_id / 16) * 8 + (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                       SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                             AB_SMEM_STRIDE]);

        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    A_smem_lane_addr);
    }

#pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&smem[B_smem_idx + lane_id % 8]
                                          [(((lane_id / 8) % 2) * 8 + (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                          SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                           AB_SMEM_STRIDE]);

        LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
    }

#pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        smem_store_idx = (smem_store_idx + 1) % K_STAGE;
        smem_store_off = smem_store_idx * smem_stage_off;

        A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters / CHUNK_K; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters / CHUNK_K; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;

#pragma unroll
        for (size_t i = (CHUNK_K - 1) * A_smem_iters / CHUNK_K; i < A_smem_iters; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

#pragma unroll
        for (size_t i = (CHUNK_K - 1) * B_smem_iters / CHUNK_K; i < B_smem_iters; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(1);

        __syncthreads();

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr =
                __cvta_generic_to_shared(&smem[A_smem_idx + lane_id % 16]
                                              [((lane_id / 16) * 8 + (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                         SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                               AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
    }

#pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(((k_step + 1) % CHUNK_K) * MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(((k_step + 1) % CHUNK_K) * MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;

            CP_ASYNC_WAIT_GROUP(0);

            __syncthreads();
        }
    }

#pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(k_step * MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

            HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                      RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], RB[reg_store_idx][j_s][0],
                      RB[reg_store_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *lane_ptr0 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;
            half *lane_ptr1 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4 + 8) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;

            *((uint32_t *)(lane_ptr0)) = RC[i][j][0];
            *((uint32_t *)(lane_ptr1)) = RC[i][j][1];
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < MMA_M; ++i) {
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
    }
}

size_t initMmaAsyncStage3() {
    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half) * K_STAGE,
                                    BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    printf("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    cudaFuncSetAttribute(mmaAsyncStage3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);

    return smem_max_size;
}

void mmaAsyncStage4(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaAsyncStage3();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    mmaAsyncStage3Kernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);

}
template<typename T>
__global__ void gemm(T *A, T *B, T *D, int M, int N, int K){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ty >= M || tx >= N){
        return;
    }
    T sum = 0.f;
    for(int i=0; i< K; ++i){
        sum += A[ty * K + i] * B[tx * K + i];
    }
    D[ty * N + tx] = sum;
}
template<typename T>
__global__ void compare_result(T *A, T *B, int M, int N){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ty >= M || tx >= N){
        return;
    }
    if(A[ty * N + tx] != B[ty * N + tx]){
        printf(" Error! %f \n",__half2float(A[ty * N + tx] - B[ty * N + tx]) );
    }
}
void col_major_to_row_major(const half* D_col_major, half* D_row_major, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            D_row_major[i * N + j] = D_col_major[j * M + i];
        }
    }
}
void rowMajorToColumnMajor(const half* rowMajorMatrix, half* columnMajorMatrix, int rows, int cols) {
    // 将行主序矩阵转换为列主序矩阵
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            columnMajorMatrix[j * rows + i] = rowMajorMatrix[i * cols + j];
        }
    }
}
int main(){
    const int M = 4096;
    const int N = 2048;
    const int K = 4096;
    using T = half;

    std::unique_ptr<T[]> h_a = std::make_unique<T[]>(M * K);
    std::unique_ptr<T[]> h_b = std::make_unique<T[]>(N * K);
    T* h_c,*h_d ;
    T* A ,*B ,*C ,*D ;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f,2.0f);
    
    checkRuntime(cudaMalloc((void**)&A,M*K*sizeof(T)));
    checkRuntime(cudaMalloc((void**)&B,K*N*sizeof(T)));
    checkRuntime(cudaMalloc((void**)&C,M*N*sizeof(T)));
    checkRuntime(cudaMalloc((void**)&D,M*N*sizeof(T)));
    checkRuntime(cudaMallocHost((void**)&h_c,M*N*sizeof(T)));
    checkRuntime(cudaMallocHost((void**)&h_d,M*N*sizeof(T)));
    for(int i=0;i<M;++i){
        for(int j=0;j<K;++j){
            h_a[i*K+j]= T(dis(gen));
            // h_a[i*K+j]= 1;
            // printf("%f ",float(h_a[i*K+j]));
        }
        // printf(" \n");
    }

    for(int i=0;i<N;++i)
        for(int j=0;j<K;++j){
            h_b[i*K+j]= T(dis(gen));
            // h_b[i*K+j]= 1;
        }

    checkRuntime(cudaMemcpy(A,h_a.get(),M*K*sizeof(T),cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(B,h_b.get(),N*K*sizeof(T),cudaMemcpyHostToDevice));
    size_t smem_max_size = std::max(Block_M * Block_N * sizeof(T), size_t((Block_M + Block_N) * MMA_K * Stage));
    dim3 grid(16,div_ceil(M, 256), div_ceil(N, 2048));
    dim3 block(256,1,1);
    mmaAsyncStage4(A,B,C,M,N,K);
    kernelCheck();
    cudaDeviceSynchronize();
    dim3 block1(16,16,1);
    dim3 grid1(div_ceil(N, block1.x),div_ceil(M, block1.y),1);
    // gemm<T><<<grid1,block1>>>(A,B,D,M,N,K);

    checkRuntime(cudaDeviceSynchronize());
    checkRuntime(cudaMemcpyAsync(h_d,D,M*N*sizeof(T),cudaMemcpyDeviceToHost));
    checkRuntime(cudaMemcpyAsync(h_c,C,M*N*sizeof(T),cudaMemcpyDeviceToHost));
    checkRuntime(cudaMemcpy(D,h_d,M*N*sizeof(T),cudaMemcpyHostToDevice));
    // compare_result<T><<<grid1,block1>>>(C, D, M, N);
    cudaDeviceSynchronize();
    for(int i=0;i<16 ;++i){
        printf("C[%d]=%f D[%d]=%f ,A[i] = %f \n",i,float(h_c[i * N]), i,float(h_d[i * N ]),float(h_a[i]));
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;

}



// #include <cstdsize_t>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <random>
#include <memory>
#include <math.h>
#include <sys/types.h>
#include <cublas_v2.h>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, size_t line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

#define kernelCheck(op) __kernel_check(__FILE__,__LINE__)
static void __kernel_check(const char* file, const size_t line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
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
// 从全局内存拷贝到共享内存，不使用缓存也就是Lcache
#define CP_ASYNC_CG(dst,src,Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(dst), "l"(src),"n"(Bytes));
// 从全局内存拷贝到共享内存，使用缓存也就是使用L2cache，但是会引入缓存延迟，所以要求缓存命中率
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
#define Stage 5
#define Block_M 256
#define Block_N 128
#define B_ 8
#define M_ 8
#define S_ 8
// #define warp_tile_i 4
// #define warp_tile_j 8
template <typename T = half>
__global__ void mma_swizzle_kstage(const T* __restrict__  A,const T* __restrict__ B,T* __restrict__ C,
                                     size_t M, size_t N, size_t K)
{


    extern __shared__ T smem[][S_ * M_];
    const size_t C_ = MMA_K / M_; 
    const size_t nums_datal_per_bankl = S_ / C_;
    
    const size_t M_tiles = div_ceil(M, MMA_M);
    size_t K_tiles = div_ceil(K, MMA_K);
    const size_t N_tiles = div_ceil(N, MMA_N);

    const size_t block_tile_i = ((blockIdx.z % 2) ? (gridDim.y - blockIdx.y -1) : (blockIdx.y))*BLOCK_COL_TILES;
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;
    
    constexpr size_t warp_tile_j = WARP_COLS / MMA_N;
    constexpr size_t warp_tile_i = WARP_ROWS / MMA_M;
    const size_t block_row_warps = Block_N / WARP_COLS;
    const size_t block_col_warps = Block_M / WARP_ROWS;

    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    if(block_tile_i >= M_tiles || block_tile_j >= N_tiles) return;
    //Calculate the starting address of block_A & block_B in global memory
    const size_t warp_off = 32 / C_;
    
    T* A_warp_ptr = (T*)(A + block_tile_i * MMA_M * K + warp_id * warp_off * K); // A行优先，则warpid * warpoff
    T* B_warp_ptr = (T*)(B + block_tile_j * MMA_N * K + warp_id * warp_off * K); // A是按行顺序取的数据，换行需要乘K，一个warp取16行，B也一样。
    //Calculate the starting address of each A & B WARP in global memory
    size_t SA_load_iters = div_ceil(Block_M, THREADS_PER_BLOCK / C_); // 总行数除以线程能一次取得行数
    size_t SB_load_iters = div_ceil(Block_N, THREADS_PER_BLOCK / C_);
    uint32_t RC[warp_tile_i][warp_tile_j][2];
#pragma unroll
    for(size_t i=0;i<warp_tile_i;++i){
#pragma unroll
        for(size_t j=0;j<warp_tile_j;++j){
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }
    size_t B_offset = Stage * SA_load_iters * 32;
    size_t i_tile_write=0;
    size_t i_tile_read=0;
    size_t istage_write = 0;
    size_t istage_read = 0;
    size_t reg_write = 0;
    size_t reg_read = 0;
#pragma unroll
    for(;istage_write < Stage-1 && i_tile_write < K_tiles; ++istage_write){
        T* A_tile_ptr = i_tile_write * MMA_K + A_warp_ptr; // 换tile读进stage，下个tile的地址是上个tile加上MMA_K，因为一个tile是Block_M x MMA_K。
        T* B_tile_ptr = i_tile_write * MMA_K + B_warp_ptr; // B也一样，虽然B小，所以B只需要线程循环一次读完，一次性读的数据与B的大小无关。
#pragma unroll
        for(size_t j=0;j < SA_load_iters; ++j){     
            T* A_gl_srt = A_tile_ptr + (j * WARP_PER_BLOCK * warp_off + lane_id % 16 ) * K \
                                                + (lane_id / 16) * M_;

            size_t bank_row = istage_write * 64 + j * 32 + warp_id / 4 * 16 + lane_id % 16 ; 
            size_t bank_col = ((warp_id % 4) * C_ + lane_id /16 ) ^ (bank_row & 7);
            uint32_t A_smem_dst = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
            CP_ASYNC_CG(A_smem_dst, A_gl_srt, THREAD_COPY_BYTES);
        }
#pragma unroll
        for(size_t j=0;j<SB_load_iters;++j){
            const T* B_gl_srt = B_tile_ptr + (j * WARP_PER_BLOCK * warp_off + lane_id % 16) * K \
                    + (lane_id / 16 ) * M_;

            size_t bank_row = B_offset + istage_write * 32 + j * 32 + warp_id / 4 * 16 + lane_id % 16;
            size_t bank_col = ((warp_id % 4 )* 2 + lane_id / 16) ^ (bank_row & 7);

            uint32_t B_smem_dst = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
            CP_ASYNC_CG(B_smem_dst, B_gl_srt, THREAD_COPY_BYTES);
        }
        ++i_tile_write;
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(Stage-2);
    __syncthreads();
    uint32_t RA[2][warp_tile_i][4];
    uint32_t RB[2][warp_tile_j][2];

#pragma unroll
    for(size_t i=0;i<warp_tile_i;++i){
        size_t bank_row = istage_read * 64 + (warp_id / block_row_warps) * MMA_M + (lane_id % MMA_M);
        size_t bank_col = (i * C_ + lane_id / MMA_M) ^ (bank_row & 7);
        uint32_t a_smem_srt = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
        LDMATRIX_X4(RA[reg_write][i][0], RA[reg_write][i][1], RA[reg_write][i][2], RA[reg_write][i][3], a_smem_srt);
    }
#pragma unroll    
    for(size_t i=0;i<warp_tile_j;++i){
        size_t bank_row = B_offset + istage_read*32 + (warp_id % block_row_warps) * MMA_K + i / 2 * 8 + lane_id % 8;
        size_t bank_col = (i / 2 * 2 + (lane_id / 8) % 2) ^ (bank_row & 7);
        uint32_t b_smem_srt = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
        LDMATRIX_X2(RB[reg_write][i][0], RB[reg_write][i][1],  b_smem_srt);
    }
    reg_write ^= 1;
    ++istage_read;
    ++i_tile_read;
#pragma unroll   
    for(size_t i=0;i<K_tiles;++i){
        
        if(i_tile_write < K_tiles){
            T* A_tile_ptr = i_tile_write * MMA_K + A_warp_ptr; 
            T* B_tile_ptr = i_tile_write * MMA_K + B_warp_ptr;
#pragma unroll
            for(size_t j=0;j < SA_load_iters; ++j){     
                T* A_gl_srt = A_tile_ptr + (j * WARP_PER_BLOCK * warp_off + lane_id % 16 ) * K \
                                                    + (lane_id / 16) * M_;

                size_t bank_row = istage_write * 64 + j * 32 + warp_id / 4 * 16 + lane_id % 16 ; 
                size_t bank_col = ((warp_id % 4) * C_ + lane_id /16 ) ^ (bank_row & 7);
                uint32_t A_smem_dst = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
                CP_ASYNC_CG(A_smem_dst, A_gl_srt, THREAD_COPY_BYTES);
            }
#pragma unroll
            for(size_t j=0;j<SB_load_iters;++j){
                T* B_gl_srt = B_tile_ptr + (j * WARP_PER_BLOCK * warp_off + lane_id % 16) * K \
                        + (lane_id / 16 ) * M_;

                size_t bank_row = B_offset + istage_write * 32 + j * 32 + warp_id / 4 * 16 + lane_id % 16;
                size_t bank_col = ((warp_id % 4 )* 2 + lane_id / 16) ^ (bank_row & 7);

                uint32_t B_smem_dst = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
                CP_ASYNC_CG(B_smem_dst, B_gl_srt, THREAD_COPY_BYTES);
            }
            ++i_tile_write;
            istage_write = (istage_write+1) % Stage;
            CP_ASYNC_COMMIT_GROUP();
        }
        if(i_tile_read < K_tiles){
#pragma unroll
            for(size_t i=0;i<warp_tile_i;++i){
                size_t bank_row = istage_read * 64 + (warp_id / block_row_warps) * MMA_M + (lane_id % MMA_M);
                size_t bank_col = (i * C_ + lane_id / MMA_M) ^ (bank_row & 7);
                uint32_t a_smem_srt = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
                LDMATRIX_X4(RA[reg_write][i][0], RA[reg_write][i][1], RA[reg_write][i][2], RA[reg_write][i][3], a_smem_srt);
            }
#pragma unroll    
            for(size_t i=0;i<warp_tile_j;++i){
                size_t bank_row = B_offset + istage_read*32 + (warp_id % block_row_warps) * MMA_K + i / 2 * 8 + lane_id % 8;
                size_t bank_col = (i / 2 * 2 + (lane_id / 8) % 2) ^ (bank_row & 7);
                uint32_t b_smem_srt = __cvta_generic_to_shared(&smem[bank_row][bank_col * M_]);
                LDMATRIX_X2(RB[reg_write][i][0], RB[reg_write][i][1],  b_smem_srt);
            }
            reg_write ^= 1;
            istage_read = (istage_read + 1) % Stage;
            ++i_tile_read;
        }

#pragma unroll
        for(size_t j =0;j< warp_tile_i;++j){
#pragma unroll
            for(size_t k=0;k< warp_tile_j;++k){
                size_t k_t = (j % 2) ? (warp_tile_j - k -1) : k;
                HMMA16816(RC[j][k_t][0], RC[j][k_t][1], \
                RA[reg_read][j][0], RA[reg_read][j][1], RA[reg_read][j][2], RA[reg_read][j][3], \
                RB[reg_read][k_t][0], RB[reg_read][k_t][1], \
                RC[j][k_t][0], RC[j][k_t][1]);
            }
        }
        reg_read ^=1;
            // load smem data to regs

        CP_ASYNC_WAIT_GROUP(Stage-2);
        __syncthreads();
    }

#pragma unroll
    for(size_t i=0;i<warp_tile_i; ++i) {
        for(size_t j = 0; j< warp_tile_j; ++j){
            size_t row0 = warp_id *  WARP_ROWS + i * MMA_M  + lane_id / 4;
            size_t row1 = row0 + 8;
            size_t col = j ^ (row0 & 7);
            *((uint32_t*)(&smem[row0][col * M_ + (lane_id % 4) * 2])) = RC[i][j][0];
            *((uint32_t*)(&smem[row1][col * M_ + (lane_id % 4) * 2])) = RC[i][j][1];
        }
    }
    __syncthreads();
#pragma unroll
    for(size_t i=0; i < MMA_N; ++i){ // store 32 rows data 8 warp ，so store 2 times to reshape 32x128shape，jump to load
        size_t row0 = i * WARP_ROWS + warp_id * 4 + lane_id / 8;
        size_t col = (lane_id % 8) ^ (row0 & 7);
        size_t row1 = row0 + 32;
        size_t global_row = i / 2 * WARP_ROWS + warp_id * 4 + lane_id / 8;
        size_t global_col = (i % 2) * 8 + (col ^ (row0 & 7)) ;
        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x==0) printf("over");
        // if(global_col >= 16) printf("over");
        *((int*)(C + block_tile_i * MMA_M * N + block_tile_j * MMA_N + global_row * N + global_col * M_)) = *((int*)(&smem[row0][col * M_]));
        *((int*)(C + block_tile_i * MMA_M * N + block_tile_j * MMA_N +  (global_row + 32) * N + global_col * M_)) = *(int*)(&smem[row1][col * M_]);
    }
}

template<typename T>
__global__ void gemm(T *A, T *B, T *D, size_t M, size_t N, size_t K){
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ty >= M || tx >= N){
        return;
    }
    T sum = 0.f;
    for(size_t i=0; i< K; ++i){
        sum += A[ty * K + i] * B[tx * K + i];
    }
    D[ty * N + tx] = sum;
}
template<typename T>
__global__ void compare_result(T *A, T *B, size_t M, size_t N){
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ty >= M || tx >= N){
        return;
    }
    if(A[ty * N + tx] != B[ty * N + tx]){
        printf(" Error! %f \n",__half2float(A[ty * N + tx] - B[ty * N + tx]) );
    }
}
void transposeWithBackup(half* matrix, size_t rows, size_t cols) {
    // 申请空间保存原来的矩阵
    half* backupMatrix = new half[rows * cols];

    // 复制原矩阵内容到备份矩阵
    for (size_t i = 0; i < rows * cols; ++i) {
        backupMatrix[i] = matrix[i];
    }

    // 转置矩阵
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[j * rows + i] = backupMatrix[i * cols + j];
        }
    }

    // 释放备份矩阵的空间
    delete[] backupMatrix;
}

void rowMajorToColumnMajor(const half* rowMajorMatrix, half* columnMajorMatrix, size_t rows, size_t cols) {
    // 将行主序矩阵转换为列主序矩阵
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            columnMajorMatrix[j * rows + i] = rowMajorMatrix[i * cols + j];
        }
    }
}
int main(){
    const size_t M = 4096;
    const size_t N = 2048;
    const size_t K = 4096;
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
    for(size_t i=0;i<M;++i){
        for(size_t j=0;j<K;++j){
            h_a[i*K+j]= T(dis(gen));
            // h_a[i*K+j]= 1;
            // printf("%f ",float(h_a[i*K+j]));
        }
        // printf(" \n");
    }

    for(size_t i=0;i<N;++i)
        for(size_t j=0;j<K;++j){
            h_b[i*K+j]= T(dis(gen));
            // h_b[i*K+j]= 1;
        }

    checkRuntime(cudaMemcpy(A,h_a.get(),M*K*sizeof(T),cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(B,h_b.get(),N*K*sizeof(T),cudaMemcpyHostToDevice));
    size_t smem_max_size = std::max(Block_M * Block_N * sizeof(T), size_t((Block_M + Block_N) * MMA_K * Stage));
    dim3 grid(16,div_ceil(M, 256), div_ceil(N, 2048));
    dim3 block(256,1,1);
    cudaFuncSetAttribute(mma_swizzle_kstage<>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);
    mma_swizzle_kstage<T><<< grid,block,smem_max_size>>> (A,B,C,M,N,K);
    kernelCheck();
    cudaDeviceSynchronize();
    dim3 block1(16,16,1);
    dim3 grid1(div_ceil(N, block1.x),div_ceil(M, block1.y),1);
    // gemm<T><<<grid1,block1>>>(A,B,D,M,N,K);
    // cublasHandle_t handle;
    // cublasCreate(&handle);

    // // cuBLAS 矩阵乘法参数：
    // half alpha = __float2half(1.0f);  // alpha 为 1.0f
    // half beta = __float2half(0.0f);   // beta 为 0.0f
    // cublasHgemm(handle,
    //             CUBLAS_OP_T, CUBLAS_OP_N,  // 不转置 A 和 B
    //             M, N, K,                  // 矩阵 A (MxK) 和 B (KxN)，结果 C (MxN)
    //             &alpha, A, M, B, K,   // 矩阵 A 和 B 的维度
    //             &beta, D, M);
    // checkRuntime(cudaMemcpyAsync(h_d,D,M*N*sizeof(T),cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());
    checkRuntime(cudaMemcpyAsync(h_d,D,M*N*sizeof(T),cudaMemcpyDeviceToHost));
    checkRuntime(cudaMemcpyAsync(h_c,C,M*N*sizeof(T),cudaMemcpyDeviceToHost));
    checkRuntime(cudaMemcpy(D,h_d,M*N*sizeof(T),cudaMemcpyHostToDevice));
    // checkRuntime(cudaDeviceSynchronize());

    // transposeWithBackup(h_d, M, N);
    // compare_result<T><<<grid1,block1>>>(C, D, M, N);
    for(size_t i=0;i<16 ;++i){
        printf("C[%d]=%f D[%d]=%f ,A[i] = %f \n",i,float(h_c[i * N +5]), i,float(h_d[i * N+5]),float(h_a[i]));
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;

}

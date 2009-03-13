/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include "particles_kernel.cuh"
#include "particles_kernel.cu"
#include "radixsort.cu"

extern "C"
{

void checkCUDA()
{   
    CUT_DEVICE_INIT();
}

void allocateArray(void **devPtr, size_t size)
{
    CUDA_SAFE_CALL(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
    CUDA_SAFE_CALL(cudaFree(devPtr));
}

void threadSync()
{
    CUDA_SAFE_CALL(cudaThreadSynchronize());
}

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size)
{   
    if (vbo)
        CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&device, vbo));
    CUDA_SAFE_CALL(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    if (vbo)
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
    CUDA_SAFE_CALL(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo)
{
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));
}

void unregisterGLBufferObject(uint vbo)
{
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo));
}

void 
integrateSystem(uint vboOldPos, uint vboNewPos, 
                float* oldVel, float* newVel, 
                float deltaTime,
                float damping,
                float particleRadius,
                float gravity,
                int numBodies)
{
    int numThreads = min(256, numBodies);
    int numBlocks = (int) ceil(numBodies / (float) numThreads);

    float *oldPos, *newPos;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldPos, vboOldPos));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&newPos, vboNewPos));

    // execute the kernel
    integrate<<< numBlocks, numThreads >>>((float4*)newPos, (float4*)newVel,
                                           (float4*)oldPos, (float4*)oldVel,
                                           deltaTime,
                                           damping,
                                           particleRadius,
                                           gravity
                                           );
    
    // check if kernel invocation generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboOldPos));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboNewPos));
}

void 
updateGrid(uint    vboPos, 
           uint*   gridCounters,
           uint*   gridCells,
           int     gridSize[3],
           float   cellSize[3],
           float   worldOrigin[3],
           int     maxParticlesPerCell,
           int     numBodies
           )
{
    int numThreads = min(256, numBodies);
    int numBlocks = (int) ceil(numBodies / (float) numThreads);

    float *pos;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&pos, vboPos));

    CUDA_SAFE_CALL(cudaMemset(gridCounters, 0, gridSize[0]*gridSize[1]*gridSize[2]*sizeof(uint)));

    // execute the kernel
    updateGridD<<< numBlocks, numThreads >>>((float4 *) pos,
                                             gridCounters,
                                             gridCells,
                                             make_uint3(gridSize[0], gridSize[1], gridSize[2]),
                                             make_float3(cellSize[0], cellSize[1], cellSize[2]),
                                             make_float3(worldOrigin[0], worldOrigin[1], worldOrigin[2]),
                                             maxParticlesPerCell
                                             );
    
    // check if kernel invocation generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboPos));
}


void 
calcHash(uint    vboPos, 
         uint*   particleHash,
         uint    gridSize[3],
         float   cellSize[3],
         float   worldOrigin[3],
         int     numBodies
         )
{
    int numThreads = min(256, numBodies);
    int numBlocks = (int) ceil(numBodies / (float) numThreads);

    float *pos;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&pos, vboPos));

    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>((float4 *) pos,
                                           (uint2 *) particleHash,
                                           make_uint3(gridSize[0], gridSize[1], gridSize[2]),
                                           make_float3(cellSize[0], cellSize[1], cellSize[2]),
                                           make_float3(worldOrigin[0], worldOrigin[1], worldOrigin[2])
                                           );
    
    // check if kernel invocation generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboPos));
}

void 
findCellStart(uint* particleHash,
              uint* cellStart,
              uint numBodies,
              uint numGridCells)
{
    // scatter method
    int numThreads = 256;
    uint numBlocks = (uint) ceil(numBodies / (float) numThreads);

    CUDA_SAFE_CALL(cudaMemset(cellStart, 0xffffffff, numGridCells*sizeof(uint)));

    findCellStartD<<< numBlocks, numThreads >>>((uint2 *) particleHash,
                                                cellStart,
                                                numBodies,
                                                numGridCells);
}

void 
reorderData(uint* particleHash,
            uint vboOldPos,
            float* oldVel,
            float* sortedPos,
            float* sortedVel,
            uint numBodies)
{
    int numThreads = 256;
    uint numBlocks = (uint) ceil(numBodies / (float) numThreads);

    float *oldPos;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldPos, vboOldPos));

#if USE_TEX
    CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numBodies*sizeof(float4)));
    CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numBodies*sizeof(float4)));
#endif

    reorderDataD<<< numBlocks, numThreads >>>((uint2 *) particleHash,
                                              (float4 *) oldPos,
                                              (float4 *) oldVel,
                                              (float4 *) sortedPos,
                                              (float4 *) sortedVel);

#if USE_TEX
    CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
    CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
#endif

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboOldPos));
}

void
collide(uint    vboOldPos, uint vboNewPos,
        float*  sortedPos, float* sortedVel,
        float*  oldVel, float* newVel, 
        uint*   gridCounters,
        uint*   gridCells,
        uint*   particleHash,
        uint*   cellStart,
        uint    gridSize[3],
        float   cellSize[3],
        float   worldOrigin[3],
        int     maxParticlesPerCell,
        float   particleRadius,
        uint    numBodies,
        float*  colliderPos,
        float   colliderRadius,
        float   spring,
        float   damping,
        float   sheer,
        float   attraction
        )
{
    float4 *oldPos, *newPos;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldPos, vboOldPos));
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&newPos, vboNewPos));

#if USE_TEX

#if USE_SORT
    // use sorted arrays
    CUDA_SAFE_CALL(cudaBindTexture(0, posTex, sortedPos, numBodies*sizeof(float4)));
    CUDA_SAFE_CALL(cudaBindTexture(0, velTex, sortedVel, numBodies*sizeof(float4)));

    CUDA_SAFE_CALL(cudaBindTexture(0, particleHashTex, particleHash, numBodies*sizeof(uint2)));
    CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTex, cellStart, gridSize[0]*gridSize[1]*gridSize[2]*sizeof(uint)));
#else

    CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numBodies*sizeof(float4)));
    CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numBodies*sizeof(float4)));

    CUDA_SAFE_CALL(cudaBindTexture(0, gridCountersTex, gridCounters, gridSize[0]*gridSize[1]*gridSize[2]*sizeof(uint)));
    CUDA_SAFE_CALL(cudaBindTexture(0, gridCellsTex, gridCells, gridSize[0]*gridSize[1]*gridSize[2]*maxParticlesPerCell*sizeof(uint)));
#endif

#endif

    // thread per particle
    int numThreads = min(BLOCKDIM, numBodies);
    int numBlocks = (int) ceil(numBodies / (float) numThreads);

    // execute the kernel
    collideD<<< numBlocks, numThreads >>>((float4*)newPos, (float4*)newVel,
#if USE_SORT
                                          (float4*)sortedPos, (float4*)sortedVel,
                                          (uint2 *) particleHash,
                                          cellStart,
#else
                                          (float4*)oldPos, (float4*)oldVel,
                                          gridCounters,
                                          gridCells,
#endif
                                          make_uint3(gridSize[0], gridSize[1], gridSize[2]),
                                          make_float3(cellSize[0], cellSize[1], cellSize[2]),
                                          make_float3(worldOrigin[0], worldOrigin[1], worldOrigin[2]),
                                          maxParticlesPerCell,
                                          particleRadius,
                                          numBodies,
                                          make_float4(colliderPos[0], colliderPos[1], colliderPos[2], 1.0f),
                                          colliderRadius,
                                          spring, damping, sheer, attraction
                                          );

    // check if kernel invocation generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboNewPos));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboOldPos));

#if USE_TEX
    CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
    CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

#if USE_SORT
    CUDA_SAFE_CALL(cudaUnbindTexture(particleHashTex));
    CUDA_SAFE_CALL(cudaUnbindTexture(cellStartTex));
#else
    CUDA_SAFE_CALL(cudaUnbindTexture(gridCountersTex));
    CUDA_SAFE_CALL(cudaUnbindTexture(gridCellsTex));
#endif
#endif
}

}   // extern "C"

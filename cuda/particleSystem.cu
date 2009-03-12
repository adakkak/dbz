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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_gl_interop.h>

#include "particles_kernel.cu"
#include "radixsort.cu"

extern "C"
{

void cudaInit(int argc, char **argv)
{   
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
}

void allocateArray(void **devPtr, size_t size)
{
    cutilSafeCall(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
    cutilSafeCall(cudaFree(devPtr));
}

void threadSync()
{
    cutilSafeCall(cudaThreadSynchronize());
}

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size)
{   
    if (vbo)
        cutilSafeCall(cudaGLMapBufferObject((void**)&device, vbo));

    cutilSafeCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    
    if (vbo)
        cutilSafeCall(cudaGLUnmapBufferObject(vbo));
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
    cutilSafeCall(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo)
{
    cutilSafeCall(cudaGLRegisterBufferObject(vbo));
}

void unregisterGLBufferObject(uint vbo)
{
    cutilSafeCall(cudaGLUnregisterBufferObject(vbo));
}

void *mapGLBufferObject(uint vbo)
{
    void *ptr;
    cutilSafeCall(cudaGLMapBufferObject(&ptr, vbo));
    return ptr;
}

void unmapGLBufferObject(uint vbo)
{
    cutilSafeCall(cudaGLUnmapBufferObject(vbo));
}

void setParameters(SimParams *hostParams)
{
    // copy parameters to constant memory
    cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
}

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

void integrateSystem(float *pos,
                     float *vel,
                     float deltaTime,
                     uint numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // execute the kernel
    integrate<<< numBlocks, numThreads >>>((float4*)pos,
                                           (float4*)vel,
                                           deltaTime,
                                           numParticles);
    
    // check if kernel invocation generated an error
    cutilCheckMsg("integrate kernel execution failed");
}

void calcHash(uint*  particleHash,
              float* pos, 
              int    numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>((uint2 *) particleHash,
                                           (float4 *) pos,
                                           numParticles);
    
    // check if kernel invocation generated an error
    cutilCheckMsg("Kernel execution failed");
}

void reorderDataAndFindCellStart(uint*  cellStart,
							     uint*  cellEnd,
							     float* sortedPos,
							     float* sortedVel,
                                 uint*  particleHash,
							     float* oldPos,
							     float* oldVel,
							     uint   numParticles,
							     uint   numCells)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // set all cells to empty
	cutilSafeCall(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
    cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
    cutilSafeCall(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
#endif

    uint smemSize = sizeof(uint)*(numThreads+1);
    reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
        cellStart,
        cellEnd,
        (float4 *) sortedPos,
        (float4 *) sortedVel,
		(uint2 *)  particleHash,
        (float4 *) oldPos,
        (float4 *) oldVel,
        numParticles);
    cutilCheckMsg("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
    cutilSafeCall(cudaUnbindTexture(oldPosTex));
    cutilSafeCall(cudaUnbindTexture(oldVelTex));
#endif
}

void collide(float* newVel,
             float* sortedPos,
             float* sortedVel,
             uint*  particleHash,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells)
{
#if USE_TEX
    cutilSafeCall(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
    cutilSafeCall(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));

    // use sorted arrays
    cutilSafeCall(cudaBindTexture(0, particleHashTex, particleHash, numParticles*sizeof(uint2)));
    cutilSafeCall(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
    cutilSafeCall(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));    
#endif

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    // execute the kernel
    collideD<<< numBlocks, numThreads >>>((float4*)newVel,
                                          (float4*)sortedPos,
                                          (float4*)sortedVel,
                                          (uint2 *) particleHash,
                                          cellStart,
                                          cellEnd,
                                          numParticles);

    // check if kernel invocation generated an error
    cutilCheckMsg("Kernel execution failed");

#if USE_TEX
    cutilSafeCall(cudaUnbindTexture(oldPosTex));
    cutilSafeCall(cudaUnbindTexture(oldVelTex));

    cutilSafeCall(cudaUnbindTexture(particleHashTex));
    cutilSafeCall(cudaUnbindTexture(cellStartTex));
    cutilSafeCall(cudaUnbindTexture(cellEndTex));
#endif
}

}   // extern "C"

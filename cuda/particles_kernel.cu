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

/* 
 * Device code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

typedef unsigned int uint;

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> posTex;
texture<float4, 1, cudaReadModeElementType> velTex;

texture<uint2, 1, cudaReadModeElementType> particleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;

texture<uint, 1, cudaReadModeElementType> gridCountersTex;
texture<uint, 1, cudaReadModeElementType> gridCellsTex;
#endif

// integrate particle attributes
__global__ void
integrate(float4* newPos, float4* newVel, 
          float4* oldPos, float4* oldVel, 
          float deltaTime,
          float damping,
          float particleRadius,
          float gravity)
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float4 pos4 = oldPos[index];
    float4 vel4 = oldVel[index];
    float3 pos = make_float3(pos4);
    float3 vel = make_float3(vel4);

    float3 force = { 0.0f, gravity, 0.0f }; 

    vel += force * deltaTime;
    vel *= damping;
        
    // new position = old position + velocity * deltaTime
    pos += vel * deltaTime;

    // bounce off cube sides
    float bounceDamping = -0.5f;
    if (pos.x > 1.0f - particleRadius) { pos.x = 1.0f - particleRadius; vel.x *= bounceDamping; }
    if (pos.x < -1.0f + particleRadius) { pos.x = -1.0f + particleRadius; vel.x *= bounceDamping;}
    if (pos.y > 1.0f - particleRadius) { pos.y = 1.0f - particleRadius; vel.y *= bounceDamping; }
    if (pos.y < -1.0f + particleRadius) { pos.y = -1.0f + particleRadius; vel.y *= bounceDamping;}
    if (pos.z > 1.0f - particleRadius) { pos.z = 1.0f - particleRadius; vel.z *= bounceDamping; }
    if (pos.z < -1.0f + particleRadius) { pos.z = -1.0f + particleRadius; vel.z *= bounceDamping;}

    // store new position and velocity
    newPos[index] = make_float4(pos, pos4.w);
    newVel[index] = make_float4(vel, vel4.w);
}

// calculate position in uniform grid
__device__ int3 calcGridPos(float4 p,
                            float3 worldOrigin,
                            float3 cellSize
                            )
{
    int3 gridPos;
    gridPos.x = floor((p.x - worldOrigin.x) / cellSize.x);
    gridPos.y = floor((p.y - worldOrigin.y) / cellSize.y);
    gridPos.z = floor((p.z - worldOrigin.z) / cellSize.z);
    return gridPos;
}

// separate each of the low 10 bits of input by 2 bits
// See: p317, Real-Time Collision Detection, Christer Ericson
__device__ uint separateBy2(uint n)
{
    // n = ----------------------9876543210 : Bits initially
    // n = ------98----------------76543210 : After (1)
    // n = ------98--------7654--------3210 : After (2)
    // n = ------98----76----54----32----10 : After (3)
    // n = ----9--8--7--6--5--4--3--2--1--0 : After (4)
    n = (n ^ (n << 16)) & 0xff0000ff; // (1)
    n = (n ^ (n <<  8)) & 0x0300f00f; // (2)
    n = (n ^ (n <<  4)) & 0x030c30c3; // (3)
    n = (n ^ (n <<  2)) & 0x09249249; // (4)
    return n;
}

// convert a 3d position into a linear 1D address in Morton (Z-curve) order
// takes three 10-bit numbers and interleaves the bits into one number
__device__ uint morton3(uint3 p)
{
    // z--z--z--z--z--z--z--z--z--z-- : separateBy2(z) << 2
    // -y--y--y--y--y--y--y--y--y--y- : separateBy2(y) << 1
    // --x--x--x--x--x--x--x--x--x--x : separateBy2(x)
    // zyxzyxzyxzyxzyxzyxzyxzyxzyxzyx : Final result
    return (separateBy2(p.z) << 2) | (separateBy2(p.y) << 1) | separateBy2(p.x);
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos,
                             uint3 gridSize)
{
    gridPos.x = max(0, min(gridPos.x, gridSize.x-1));
    gridPos.y = max(0, min(gridPos.y, gridSize.y-1));
    gridPos.z = max(0, min(gridPos.z, gridSize.z-1));
    return __mul24(__mul24(gridPos.z, gridSize.y), gridSize.x) + __mul24(gridPos.y, gridSize.x) + gridPos.x;
//    return morton3(make_uint3(gridPos.x, gridPos.y, gridPos.z));
}

// add particle to cell using atomics
__device__ void addParticleToCell(int3 gridPos,
                                  uint index,
                                  uint* gridCounters,
                                  uint* gridCells,
                                  uint3 gridSize,
                                  int maxParticlesPerCell
                                  )
{
    // calculate grid hash
    uint gridHash = calcGridHash(gridPos, gridSize);

    // increment cell counter using atomics
#if defined CUDA_NO_SM_11_ATOMIC_INTRINSICS
    int counter = 0;
#else
    int counter = atomicAdd(&gridCounters[gridHash], 1); // returns previous value
    counter = min(counter, maxParticlesPerCell-1);
#endif

    // write particle index into this cell (very uncoalesced!)
    gridCells[gridHash*maxParticlesPerCell + counter] = index;
}


// update uniform grid
__global__ void
updateGridD(float4* pos,
            uint*   gridCounters,
            uint*   gridCells,
            uint3   gridSize,
            float3  cellSize,
            float3  worldOrigin,
            int     maxParticlesPerCell
            )
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(p, worldOrigin, cellSize);

    addParticleToCell(gridPos, index, gridCounters, gridCells, gridSize, maxParticlesPerCell);
}

// calculate grid hash value for each particle
__global__ void
calcHashD(float4* pos,
          uint2*  particleHash,
          uint3   gridSize,
          float3  cellSize,
          float3  worldOrigin
          )
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(p, worldOrigin, cellSize);
    uint gridHash = calcGridHash(gridPos, gridSize);

    // store grid hash and particle index
    particleHash[index] = make_uint2(gridHash, index);
}

// find start of each cell in sorted particle list by comparing with previous hash value
// one thread per particle
__global__ void
findCellStartD(uint2* particleHash,
               uint * cellStart,
               uint   numBodies,
               uint   numCells)
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint cell = particleHash[i].x;

    if (i > 0) {
        if (cell != particleHash[i-1].x) {
            cellStart[ cell ] = i;
        }
    } else {
        cellStart[ cell ] = i;
    }
}

// rearrange particle data into sorted order
__global__ void
reorderDataD(uint2*  particleHash,  // particle id sorted by hash
             float4* oldPos,
             float4* oldVel,
             float4* sortedPos, 
             float4* sortedVel
             )
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint sortedIndex = particleHash[index].y;
#if USE_TEX
	float4 pos = tex1Dfetch(posTex, sortedIndex);
    float4 vel = tex1Dfetch(velTex, sortedIndex);
#else
	float4 pos = oldPos[sortedIndex];
    float4 vel = oldVel[sortedIndex];
#endif

    sortedPos[index] = pos;
    sortedVel[index] = vel;
}

// collide two spheres using DEM method
__device__ float3 collideSpheres(float4 posA, float4 posB,
                                 float4 velA, float4 velB,
                                 float radiusA, float radiusB,
                                 float spring,
                                 float damping,
                                 float shear,
                                 float attraction
                                 )
{
	// calculate relative position
    float3 relPos;
    relPos.x = posB.x - posA.x;
    relPos.y = posB.y - posA.y;
    relPos.z = posB.z - posA.z;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);
    if (dist < collideDist) {
        float3 norm = relPos / dist;

		// relative velocity
        float3 relVel;
        relVel.x = velB.x - velA.x;
        relVel.y = velB.y - velA.y;
        relVel.z = velB.z - velA.z;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += damping*relVel;
        // tangential shear force
        force += shear*tanVel;
		// attraction
        force += attraction*relPos;
    }

    return force;
}


// collide particle with all particles in a given cell
// version using grid built with atomics
__device__
float3 collideCell(int3 gridPos,
                   uint index,
                   float4 pos,
                   float4 vel,
                   float4* oldPos, 
                   float4* oldVel,
                   uint*   gridCounters,
                   uint*   gridCells,
                   uint3   gridSize,
                   int     maxParticlesPerCell,
                   float   particleRadius,
                   float   spring,
                   float   damping,
                   float   shear,
                   float   attraction
                   )
{
    float3 force = make_float3(0.0f);

    if ((gridPos.x < 0) || (gridPos.x > gridSize.x-1) ||
        (gridPos.y < 0) || (gridPos.y > gridSize.y-1) ||
        (gridPos.z < 0) || (gridPos.z > gridSize.z-1)) {
        return force;
    }

    uint gridHash = calcGridHash(gridPos, gridSize);   
    
    // iterate over particles in this cell
#if USE_TEX
    uint particlesInCell = tex1Dfetch(gridCountersTex, gridHash);
#else
    uint particlesInCell = gridCounters[gridHash];
#endif
    particlesInCell = min(particlesInCell, maxParticlesPerCell-1);

    for(uint i=0; i<particlesInCell; i++) {
#if USE_TEX
        uint index2 = tex1Dfetch(gridCellsTex, gridHash*maxParticlesPerCell + i);
#else
        uint index2 = gridCells[gridHash*maxParticlesPerCell + i];
#endif

        if (index2 != index) {              // check not colliding with self
#if USE_TEX
	        float4 pos2 = tex1Dfetch(posTex, index2);
            float4 vel2 = tex1Dfetch(velTex, index2);
#else
            float4 pos2 = oldPos[index2];
            float4 vel2 = oldVel[index2];
#endif

            // collide two spheres
            float3 projVec = collideSpheres(pos, pos2, vel, vel2, particleRadius, particleRadius, spring, damping, shear, attraction);
            force += projVec;
        }
    }

    return force;
}


// version using sorted grid
__device__
float3 collideCell2(int3   gridPos,
                   uint    index,
                   float4  pos,
                   float4  vel,
                   float4* oldPos, 
                   float4* oldVel,
                   uint2*  particleHash,
                   uint*   cellStart,
                   uint3   gridSize,
                   int     maxParticlesPerCell,
                   float   particleRadius,
                   float   spring,
                   float   damping,
                   float   shear,
                   float   attraction
                   )
{
    float3 force = make_float3(0.0f);

    if ((gridPos.x < 0) || (gridPos.x > gridSize.x-1) ||
        (gridPos.y < 0) || (gridPos.y > gridSize.y-1) ||
        (gridPos.z < 0) || (gridPos.z > gridSize.z-1)) {
        return force;
    }

    uint gridHash = calcGridHash(gridPos, gridSize);

    // get start of bucket for this cell
#if USE_TEX
    uint bucketStart = tex1Dfetch(cellStartTex, gridHash);
#else
    uint bucketStart = cellStart[gridHash];
#endif
    if (bucketStart == 0xffffffff)
        return force;   // cell empty
 
    // iterate over particles in this cell
    for(uint i=0; i<maxParticlesPerCell; i++) {
        uint index2 = bucketStart + i;
#if USE_TEX
        uint2 cellData = tex1Dfetch(particleHashTex, index2);
#else
        uint2 cellData = particleHash[index2];
#endif
        if (cellData.x != gridHash) break;   // no longer in same bucket

        if (index2 != index) {              // check not colliding with self
#if USE_TEX
	        float4 pos2 = tex1Dfetch(posTex, index2);
            float4 vel2 = tex1Dfetch(velTex, index2);
#else
            float4 pos2 = oldPos[index2];
            float4 vel2 = oldVel[index2];
#endif

            // collide two spheres
            float3 projVec = collideSpheres(pos, pos2, vel, vel2, particleRadius, particleRadius, spring, damping, shear, attraction);
            force += projVec;
        }
    }

    return force;
}


__global__ void
collideD(float4* newPos, float4* newVel, 
         float4* oldPos, float4* oldVel, 
#if USE_SORT
         uint2*  particleHash,
         uint*   cellStart,
#else
         uint*   gridCounters,
         uint*   gridCells,
#endif
         uint3   gridSize,
         float3  cellSize,
         float3  worldOrigin,
         int     maxParticlesPerCell,
         float   particleRadius,
         uint    numBodies,
         float4  colliderPos,
         float   colliderRadius,
         float   spring,
         float   damping,
         float   shear,
         float   attraction
         )
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    // read particle data from sorted arrays
#if USE_TEX
	float4 pos = tex1Dfetch(posTex, index);
    float4 vel = tex1Dfetch(velTex, index);
#else
	float4 pos = oldPos[index];
    float4 vel = oldVel[index];
#endif

    // get address in grid
    int3 gridPos = calcGridPos(pos, worldOrigin, cellSize);

    float3 force = make_float3(0.0f);

    // examine only neighbouring cells
    for(int z=-1; z<=1; z++) {
        for(int y=-1; y<=1; y++) {
            for(int x=-1; x<=1; x++) {
#if USE_SORT
                force += collideCell2(gridPos + make_int3(x, y, z), index, pos, vel, oldPos, oldVel, particleHash, cellStart, gridSize, maxParticlesPerCell, particleRadius, spring, damping, shear, attraction);
#else
                force += collideCell(gridPos + make_int3(x, y, z), index, pos, vel, oldPos, oldVel, gridCounters, gridCells, gridSize, maxParticlesPerCell, particleRadius, spring, damping, shear, attraction);
#endif
            }
        }
    }

    float3 projVec = collideSpheres(pos, colliderPos, vel, make_float4(0.0f, 0.0f, 0.0f, 0.0f), particleRadius, colliderRadius, spring, damping, shear, 0.0f);
    force += projVec;

#if USE_SORT
    // write new velocity back to original unsorted location
    index = particleHash[index].y;
#endif
    newVel[index] = vel + make_float4(force, 0.0f);
}

#endif

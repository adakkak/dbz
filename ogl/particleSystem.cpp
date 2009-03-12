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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "radixsort.cuh"
#include "particles_kernel.cuh"

#include <cutil_inline.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_dPos(0),
    m_dVel(0),
    m_gridSize(gridSize),
    m_timer(0),
    m_solverIterations(1)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
//    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    registerGLBufferObject(vbo);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] = {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
	    { 1.0, 1.0, 0.0, },
	    { 0.0, 1.0, 0.0, },
	    { 0.0, 1.0, 1.0, },
	    { 0.0, 0.0, 1.0, },
	    { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));

    m_hParticleHash = new uint[m_numParticles*2];
    memset(m_hParticleHash, 0, m_numParticles*2*sizeof(uint));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    m_posVbo = createVBO(memSize);    
    allocateArray((void**)&m_dVel, memSize);

    allocateArray((void**)&m_dSortedPos, memSize);
    allocateArray((void**)&m_dSortedVel, memSize);

    allocateArray((void**)&m_dParticleHash[0], m_numParticles*2*sizeof(uint));
    allocateArray((void**)&m_dParticleHash[1], m_numParticles*2*sizeof(uint));
    allocateArray((void**)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void**)&m_dCellEnd, m_numGridCells*sizeof(uint));

    m_colorVBO = createVBO(m_numParticles*4*sizeof(float));

    // fill color buffer
    glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
    float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float *ptr = data;
    for(uint i=0; i<m_numParticles; i++) {
        float t = i / (float) m_numParticles;
#if 0
        *ptr++ = rand() / (float) RAND_MAX;
        *ptr++ = rand() / (float) RAND_MAX;
        *ptr++ = rand() / (float) RAND_MAX;
#else
        colorRamp(t, ptr);
        ptr+=3;
#endif
        *ptr++ = 1.0f;
    }
    glUnmapBufferARB(GL_ARRAY_BUFFER);

    cutilCheckError(cutCreateTimer(&m_timer));

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hParticleHash;
    delete m_hCellStart;
    delete m_hCellEnd;

    freeArray(m_dVel);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dParticleHash[0]);
    freeArray(m_dParticleHash[1]);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    unregisterGLBufferObject(m_posVbo);
    glDeleteBuffers(1, (const GLuint*)&m_posVbo);
    glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
}

// step the simulation
void 
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    // update constants
    setParameters(&m_params);

    float *dPos = (float *) mapGLBufferObject(m_posVbo);

    // integrate
    integrateSystem(dPos,
                    m_dVel,
                    deltaTime,
                    m_numParticles);

    // calculate grid hash
    calcHash(m_dParticleHash[0],
             dPos,
             m_numParticles);

    // sort particles based on hash
    RadixSort((KeyValuePair *) m_dParticleHash[0], (KeyValuePair *) m_dParticleHash[1], m_numParticles, 32);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(m_dCellStart,
                                m_dCellEnd,
								m_dSortedPos,
								m_dSortedVel,
                                m_dParticleHash[0],
								dPos,
								m_dVel,
								m_numParticles,
								m_numGridCells);

    unmapGLBufferObject(m_posVbo);

    // process collisions
    collide(m_dVel,
            m_dSortedPos,
            m_dSortedVel,
            m_dParticleHash[0],
            m_dCellStart,
            m_dCellEnd,
            m_numParticles,
            m_numGridCells);

}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;
    for(uint i=0; i<m_numGridCells; i++) {
        if (m_hCellStart[i] != 0xffffffff) {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];
//            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize) maxCellSize = cellSize;
        }
    }
    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, 0, m_posVbo, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);

    for(uint i=start; i<start+count; i++) {
//        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
}

float* 
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);
 
    float* hdata = 0;
    float* ddata = 0;

    unsigned int vbo = 0;

    switch (array)
    {
    default:
    case POSITION:
        hdata = m_hPos;
        ddata = m_dPos;
        vbo = m_posVbo;
        break;
    case VELOCITY:
        hdata = m_hVel;
        ddata = m_dVel;
        break;
    }

    copyArrayFromDevice(hdata, ddata, vbo, m_numParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float* data, int start, int count)
{
    assert(m_bInitialized);
 
    switch (array)
    {
    default:
    case POSITION:
        {
            unregisterGLBufferObject(m_posVbo);
            glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
            glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            registerGLBufferObject(m_posVbo);
        }
        break;
    case VELOCITY:
        copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
        break;
    }       
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);
    for(uint z=0; z<size[2]; z++) {
        for(uint y=0; y<size[1]; y++) {
            for(uint x=0; x<size[0]; x++) {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;
                if (i < numParticles) {
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+3] = 1.0f;

				    m_hVel[i*4] = 0.0f;
				    m_hVel[i*4+1] = 0.0f;
				    m_hVel[i*4+2] = 0.0f;
				    m_hVel[i*4+3] = 0.0f;
                }
            }
        }
    }
}

void
ParticleSystem::reset(ParticleConfig config)
{
	switch(config)
	{
	default:
	case CONFIG_RANDOM:
		{
			int p = 0, v = 0;
			for(uint i=0; i < m_numParticles; i++) 
			{
				float point[3];
				point[0] = frand();
				point[1] = frand();
				point[2] = frand();
				m_hPos[p++] = 2 * (point[0] - 0.5f);
				m_hPos[p++] = 2 * (point[1] - 0.5f);
				m_hPos[p++] = 2 * (point[2] - 0.5f);
				m_hPos[p++] = 1.0f; // radius
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
				m_hVel[v++] = 0.0f;
			}
		}
		break;

    case CONFIG_GRID:
        {
            float jitter = m_params.particleRadius*0.01f;
            uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
            uint gridSize[3];
            gridSize[0] = gridSize[1] = gridSize[2] = s;
            initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
        }
        break;
	}

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;
    for(int z=-r; z<=r; z++) {
        for(int y=-r; y<=r; y++) {
            for(int x=-r; x<=r; x++) {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius*0.01f;
                if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles)) {
                    m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter; 
                    m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3] = pos[3];

                    m_hVel[index*4]   = vel[0];
                    m_hVel[index*4+1] = vel[1];
                    m_hVel[index*4+2] = vel[2];
                    m_hVel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}

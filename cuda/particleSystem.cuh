extern "C"
{
void checkCUDA();

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(unsigned int vbo);
void unregisterGLBufferObject(unsigned int vbo);

void integrateSystem(uint vboOldPos, uint vboNewPos,
                     float* oldVel, float* newVel,
                     float deltaTime,
                     float damping,
                     float particleRadius,
                     float gravity,
                     int numBodies);

void updateGrid(uint vboPos, 
                uint*   gridCounters,
                uint*   gridCells,
                uint    gridSize[3],
                float   cellSize[3],
                float   worldOrigin[3],
                int     maxParticlesPerCell,
                int     numBodies);

void 
calcHash(uint    vboPos, 
         uint*   particleHash,
         uint    gridSize[3],
         float   cellSize[3],
         float   worldOrigin[3],
         int     numBodies);

void 
findCellStart(uint* particleHash,
              uint* cellStart,
              uint numBodies,
              uint numGridCells);

void 
reorderData(uint* particleHash,
            uint vboOldPos,
            float* oldVel,
            float* sortedPos,
            float* sortedVel,
            uint numBodies);

void collide(uint    vboOldPos, uint vboNewPos,
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
             float   *colliderPos,
             float   colliderRadius,
             float   spring,
             float   damping,
             float   shear,
             float   attraction);

}

extern "C"
{
void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(unsigned int vbo);
void unregisterGLBufferObject(unsigned int vbo);
void *mapGLBufferObject(uint vbo);
void unmapGLBufferObject(uint vbo);

void setParameters(SimParams *hostParams);

void integrateSystem(float *pos,
                     float *vel,
                     float deltaTime,
                     uint numParticles);

void calcHash(uint*  particleHash,
              float* pos, 
              int    numParticles);

void reorderDataAndFindCellStart(uint*  cellStart,
							     uint*  cellEnd,
							     float* sortedPos,
							     float* sortedVel,
                                 uint*  particleHash,
							     float* oldPos,
							     float* oldVel,
							     uint   numParticles,
							     uint   numCells);

void collide(float* newVel,
             float* sortedPos,
             float* sortedVel,
             uint*  particleHash,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells);

}

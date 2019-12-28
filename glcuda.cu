#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

int image_width = 256;
int image_height = 256;
unsigned int *cuda_dest_resource;
struct cudaGraphicsResource *cuda_tex_result_resource;
GLuint shDrawTex;
GLuint fbo_source;
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result
GLuint shDraw;
struct cudaGraphicsResource *cuda_tex_screen_resource;
unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;
//void registerCUDAGL() {
//    cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
//}
__global__ void process(unsigned int *canvas, int imgw)
{
    // thread indices
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // original value at location y,x
    unsigned int in = canvas[y*imgw+x];
    unsigned int out = (x - y) + in;

    canvas[y*imgw+x] = out;
}

extern "C" void launch_process(dim3 grid, dim3 block, int sbytes, unsigned int *canvas, int imgw)
{
    process<<< grid, block, sbytes >>>(canvas, imgw);

}

extern "C" {
  void generateCUDAImage() {
    printf("generate cuda image\n");

    // run the Cuda kernel
    unsigned int *out_data;

//#ifdef USE_TEXSUBIMAGE2D
//    cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0);
//    size_t num_bytes;
//    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&out_data, &num_bytes,
//                                                         cuda_pbo_dest_resource));
//    //printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n", num_bytes, size_tex_data);
//#else
    out_data = cuda_dest_resource;
//#endif
    // calculate grid size
    dim3 block(16, 16, 1);
    //dim3 block(16, 16, 1);
    //dim3 grid(image_width / block.x, image_height / block.y, 1);
    dim3 grid(16, 16, 1);
    // execute CUDA kernel
    launch_process(grid, block, 0, out_data, image_width);

    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    // 2 solutions, here :
    // - use glTexSubImage2D(), there is the potential to loose performance in possible hidden conversion
    // - map the texture and blit the result thanks to CUDA API
//#ifdef USE_TEXSUBIMAGE2D
//    cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);
//
//    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
//    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
//                    image_width, image_height,
//                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);
//    //SDK_CHECK_ERROR_GL();
//    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
//#else
    // We want to copy cuda_dest_resource data to the texture
    // map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0);

    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(unsigned char) * num_values;
    cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
//#endif
}

}

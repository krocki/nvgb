// GL frontend to GPUboy
// kmrocki @ 1/16/19
//

#include <stdio.h>
#include <sys/time.h>

unsigned int window_width = 256;
unsigned int window_height = 256;
double last_time;

double get_time() {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

int iGLUTWindowHandle = 0;          // handle to the GLUT window
int enable_cuda = 1;

#if defined(__APPLE__) || defined(MACOSX)
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
    #ifdef __linux__
    #include <GL/glx.h>
    #endif /* __linux__ */
#endif

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#define USE_TEXSUBIMAGE2D
#else
#include <GL/freeglut.h>
#endif

static int fpsCount = 0;
static int fpsLimit = 1;
extern int image_width;
extern int image_height;
extern GLuint tex_cudaResult;
extern unsigned int *cuda_dest_resource;
extern GLuint shDrawTex;  // draws a texture
extern struct cudaGraphicsResource *cuda_tex_result_resource;
extern GLuint fbo_source;
extern GLuint tex_screen;      // where we render the image
extern GLuint shDraw;
extern unsigned int size_tex_data;
extern unsigned int num_texels;
extern unsigned int num_values;
extern struct cudaGraphicsResource *cuda_tex_screen_resource;

GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src);

void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint *tex);

#define REFRESH_DELAY 10 //ms
const char *window_name = "GPUboy";

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void timerEvent(int);

const GLenum fbo_targets[] =
{
    GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
    GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
};

// this one is in use
static const char *glsl_drawtex_vertshader_src =
    "void main(void)\n"
    "{\n"
    " gl_Position = gl_Vertex;\n"
    " gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
    "}\n";

static const char *glsl_drawtex_fragshader_src =
    "#version 130\n"
    "uniform usampler2D texImage;\n"
    "void main()\n"
    "{\n"
    "   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
    "   gl_FragColor = c / 255.0;\n"
    "}\n";

static const char *glsl_draw_fragshader_src =
    //WARNING: seems like the gl_FragColor doesn't want to output >1 colors...
    //you need version 1.3 so you can define a uvec4 output...
    //but MacOSX complains about not supporting 1.3 !!
    // for now, the mode where we use RGBA8UI may not work properly for Apple : only RGBA16F works (default)
#if defined(__APPLE__) || defined(MACOSX)
    "void main()\n"
    "{"
    "  gl_FragColor = vec4(gl_Color * 255.0);\n"
    "}\n";
#else
    "#version 130\n"
    "out uvec4 FragColor;\n"
    "void main()\n"
    "{"
    "  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
    "}\n";
#endif
//
// copy image and process using CUDA
int initGL(int* argc, char **argv) {

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("window");

    // initialize necessary OpenGL extensions
    //if (!isGLVersionSupported (2,0) ||
    //    !areGLExtensionsSupported (
    //        "GL_ARB_pixel_buffer_object "
    //        "GL_EXT_framebuffer_object"
    //    ))
    //{
    //    printf("ERROR: Support for necessary OpenGL extensions missing.");
    //    return 0;
    //}

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1f, 10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

    return 1;
}

void main(int argc, char **argv) {

    if (!initGL(&argc, argv)) {
      printf("could not init openGL !\n"); return;
    }

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    initGLBuffers();
    initCUDABuffers();

    glutMainLoop();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case (27) : // ESC
          exit(0);
          //Cleanup(EXIT_SUCCESS);
          break;

        case ' ':
            enable_cuda ^= 1;
            printf("CUDA enabled = %d\n", enable_cuda);
            break;

    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
}

void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    // if the texture is a 8 bits UI, scale the fetch with a GLSL shader
    glUseProgram(shDrawTex);
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0); // texture unit 0 to "texImage"

    //glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

    glUseProgram(0);
}

GLuint tex_cudaResult;  // where we will copy the CUDA result

void display()
{

    if (enable_cuda)
    {
        generateCUDAImage();
        //if (init) init=0;
        displayImage(tex_cudaResult);
    }

    cudaDeviceSynchronize();

    // flip backbuffer
    glutSwapBuffers();
    // Update fps counter, fps/title display and log
    char cTitle[256];
    double cur_time = get_time();
    double frame_time = cur_time - last_time; last_time = cur_time;
    float fps = 1.0f / frame_time;
    sprintf(cTitle, "%.1f fps (%d x %d)", fps, window_width, window_height);
    //printf("%s\n", cTitle);
    glutSetWindowTitle(cTitle);
}

void timerEvent(int value) {

    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void cleanup() {

    cudaFree(cuda_dest_resource);
    deleteTexture(&tex_screen);
    deleteTexture(&tex_cudaResult);

    if (iGLUTWindowHandle) { glutDestroyWindow(iGLUTWindowHandle); }
}

void initCUDABuffers()
{
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    cudaMalloc((void **)&cuda_dest_resource, size_tex_data);
    //checkCudaErrors(cudaHostAlloc((void**)&cuda_dest_resource, size_tex_data, ));
}

void initGLBuffers()
{
    // create texture that will receive the result of CUDA
    createTextureDst(&tex_cudaResult, image_width, image_height);
    // load shader programs
    shDraw = compileGLSLprogram(NULL, glsl_draw_fragshader_src);
    shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);
}

GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
    GLuint v, f, p = 0;

    p = glCreateProgram();

    if (vertex_shader_src)
    {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vertex_shader_src, NULL);
        glCompileShader(v);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(v, 256, NULL, temp);
            printf("Vtx Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(v);
            return 0;
        }
        else
        {
            glAttachShader(p,v);
        }
    }

    if (fragment_shader_src)
    {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fragment_shader_src, NULL);
        glCompileShader(f);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(f, 256, NULL, temp);
            printf("frag Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(f);
            return 0;
        }
        else
        {
            glAttachShader(p,f);
        }
    }

    glLinkProgram(p);

    int infologLength = 0;
    int charsWritten  = 0;

    glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

    if (infologLength > 0)
    {
        char *infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
        printf("Shader compilation error: %s\n", infoLog);
        free(infoLog);
    }

    return p;
}
void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    // register this texture with CUDA
    cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult, GL_TEXTURE_2D, 0);
}
void deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex); *tex = 0;
}

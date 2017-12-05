#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <assert.h>

#include "CL/cl.h"
using namespace cv;

std::string programSource;

extern "C"
JNIEXPORT jint
JNICALL
Java_com_cloudream_myapplication_MainActivity_OpenFile(
        JNIEnv *env,
        jobject /* this */,
        jobject assetManager ) {

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager );
    AAsset* asset = AAssetManager_open(mgr, "kernel.cl", AASSET_MODE_UNKNOWN);
    assert(asset);

    auto length = AAsset_getLength(asset);
    programSource.resize(length);
    memcpy(&programSource[0], AAsset_getBuffer(asset), length);
    AAsset_close(asset);
    return 0;

}

const char* programSrc =
        "__kernel void simpleMultiply( \n"
        "    __global float* outputC, \n"
        "    int widthA,\n"
        "    int heightA,\n"
        "    int widthB,\n"
        "    int heightB,\n"
        "    __global float* inputA,\n"
        "    __global float* inputB)\n"
"{\n"

    "   int row = get_global_id(1);\n"
    "   int col = get_global_id(0);\n"
    "   float sum = 0.0f;\n"
    "   for(int i = 0; i < widthA; i++)\n"
    "   {\n"
            "  sum += inputA[row * widthA + i] * inputB[i * widthB + col];\n"
    "   }\n"
    "   outputC[row * widthB + col] = sum;\n"
"}";


extern "C"
JNIEXPORT jstring
JNICALL
Java_com_cloudream_myapplication_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    std::ofstream out("./out.txt");

    int len = strlen(programSrc);
    auto size = programSource.size();
    for (int i = 0; i < programSource.size(); ++i) {
        if (programSource[i] != programSrc[i])
        {
            printf("hello world");
        }
    }

    const int widthA = 640;
    const int heightA = 640;
    const int widthB = 640;
    const int heightB = 640;
    const int widthC = widthB;
    const int heightC = heightA;

    float *A = NULL;
    float *B = NULL;
    float *C = NULL;


    size_t dataSizeA = sizeof(float) * widthA * heightA;
    size_t dataSizeB = sizeof(float) * widthB * heightB;
    size_t dataSizeC = sizeof(float) * heightA * widthB;

    A = (float *)malloc(dataSizeA);
    B = (float *)malloc(dataSizeB);
    C = (float*)malloc(dataSizeC);

    for (int i = 0; i < widthA *heightA; ++i)
    {
        if (i % 3 == 0)
        {
            A[i] = 2.5f;
        }
        else if (i % 3 == 1)
        {
            A[i] = 1.5f;
        }
        else
        {
            A[i] = 2.0f;
        }
    }

    for (int i = 0; i < widthB * heightB; ++i) {
        if (i % 3 == 0)
        {
            B[i] = 2.5f;
        }
        else if (i % 3 == 1)
        {
            B[i] = 1.5f;
        }
        else
        {
            B[i] = 2.0f;
        }

    }

    clock_t start, end;
    start = clock();
    for (int i = 0; i < heightA; i++)
    {
        for (int j = 0; j < widthB; j++)
        {
            C[i * widthB + j] = 0;
            for (int k = 0; k < widthA; k++)
            {
                C[i * widthB + j] += A[i * widthA + k] * B[k * widthB + j];
            }
        }
    }
    end = clock();
    std::string during2 = std::to_string((double) (end - start) / CLOCKS_PER_SEC);
    float *D = (float*) malloc(dataSizeC);
    memcpy(D, C, dataSizeC);



    //use the first platform
    cl_int ciErrNum;
    cl_platform_id platform;
    ciErrNum = clGetPlatformIDs(1, &platform, NULL);
    if (ciErrNum != CL_SUCCESS){
        printf("function clGetPlatformIDs goes wrong\n");
    }

    //use the first device

    cl_device_id device;
    ciErrNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    if (ciErrNum != CL_SUCCESS){
        printf("function clGetDeviceIDs goes wrong\n");
    }

    cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

    //create the context
    cl_context ctx = clCreateContext(cps, 1, &device, NULL, NULL, &ciErrNum);

    if (ciErrNum != CL_SUCCESS){
        printf("function clCreateContext goes wrong");
    }

    //create the command queue

    cl_command_queue myqueue = clCreateCommandQueue(ctx, device, 0, &ciErrNum);

    if (ciErrNum != CL_SUCCESS){
        printf("function clCreateCommandQueue goes wrong");
    }
    start = clock();
    //declare the buffer and transimit it
    cl_mem bufferA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, dataSizeA, NULL, &ciErrNum);
    if (ciErrNum == CL_SUCCESS){
        printf("buffer A created successfully!\n");
    }


    ciErrNum = clEnqueueWriteBuffer(myqueue, bufferA, CL_FALSE, 0, dataSizeA, (void *)A, 0, NULL, NULL);
    end = clock();
    if (ciErrNum == CL_SUCCESS){
        printf("bufferA has transformed successfully!\n");
    }

    cl_mem bufferB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, dataSizeB, NULL, &ciErrNum);
    ciErrNum = clEnqueueWriteBuffer(myqueue, bufferB, CL_FALSE, 0, dataSizeB, (void *)B, 0, NULL, NULL);
    if (ciErrNum == CL_SUCCESS){
        printf("bufferB has transformed successfully!\n");
    }

    cl_mem bufferC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, dataSizeC, NULL, &ciErrNum);

    if (ciErrNum == CL_SUCCESS){
        printf("buffer C has created successfully!\n");
    }
    const char * src = programSource.c_str();
    //compile and build kernel
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &ciErrNum);
    if (ciErrNum == CL_SUCCESS){
        printf("create program successfully!\n");
    }
    if (ciErrNum != CL_SUCCESS){
        printf("create program wrong\n");
    }

    ciErrNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum == CL_SUCCESS){
        printf("build program successfully!\n");
    }
    else
    {
        char *buff_erro;
        cl_int errcode = 0;
        size_t build_log_len = 0;
        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-1);
        }

        buff_erro = (char*)malloc(build_log_len);
        if (!buff_erro) {
            printf("malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }
        buff_erro[build_log_len] = '\0';
        fprintf(stderr, "Build log: \n%s\n", buff_erro); //Be careful with  the fprint
        free(buff_erro);
        fprintf(stderr, "clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }

    cl_kernel mykernel = clCreateKernel(program, "simpleMultiply", &ciErrNum);
    if (ciErrNum == CL_SUCCESS){
        printf("create kernel successfully!\n");
    }

    if (ciErrNum != CL_SUCCESS){
        printf("create kernel wrong\n");
    }

    //set the kernel arguments
    ciErrNum = clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&bufferC);
    ciErrNum |= clSetKernelArg(mykernel, 1, sizeof(cl_int), (void *)&widthA);
    ciErrNum |= clSetKernelArg(mykernel, 2, sizeof(cl_int), (void *)&heightA);
    ciErrNum |= clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&widthB);
    ciErrNum |= clSetKernelArg(mykernel, 4, sizeof(cl_int), (void *)&heightB);
    ciErrNum |= clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void *)&bufferA);
    ciErrNum |= clSetKernelArg(mykernel, 6, sizeof(cl_mem), (void *)&bufferB);
    printf("set kernel arguments successfully!\n");
    if (ciErrNum != CL_SUCCESS){
        printf("set kernel arguments wrong\n");
    }

    //size_t localws[2] = {16,16};
    size_t globalws[2] = { widthC, heightC};

    //execute the kernel

    ciErrNum = clEnqueueNDRangeKernel(myqueue, mykernel, 2, NULL, globalws, NULL, 0, NULL, NULL);
    clFinish(myqueue);

    if (ciErrNum == CL_SUCCESS){
        printf("execute the kernel successfully!\n");
    }

    if (ciErrNum != CL_SUCCESS){
        printf("execute the kernel wrong\n");
    }
    //Read the output

    ciErrNum = clEnqueueReadBuffer(myqueue, bufferC, CL_TRUE, 0, dataSizeC, (void *)C, 0, NULL, NULL);

    std::string during = std::to_string((double)(end - start) / CLOCKS_PER_SEC);

    std::string result = "true";
    for(int i = 0; i < widthC * heightC; i++)
    {
        if(C[i] != D[i])
        {
            result = "false";
            break;
        }
    }
//    std::string ps;
//    for (int i = 0; i < heightC; ++i)
//    {
//        for (int j = 0; j < widthC; ++j)
//        {
//            ps += std::to_string(C[i * widthC + j]);
//        }
////        out << std::endl;
//    }
    clReleaseKernel(mykernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(myqueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(ctx);

    free(A);
    free(B);
    free(C);
    free(platform);
    free(device);

//    std::string hello = "Hello from C++";
    return env->NewStringUTF((during + "       " + during2 + result).c_str());
}

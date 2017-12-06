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
#include <chrono>

#include "CL/cl.h"
using namespace cv;

std::string programSource;
cv::Mat img;
std::string path;
extern "C"
JNIEXPORT jint
JNICALL
Java_com_cloudream_myapplication_MainActivity_OpenFile(
        JNIEnv *env,
        jobject /* this */,
        jobject assetManager,
        jstring fileUrl) {

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager );
    AAsset* asset = AAssetManager_open(mgr, "kernel.cl", AASSET_MODE_UNKNOWN);
    assert(asset);

    auto length = AAsset_getLength(asset);
    programSource.resize(length);
    memcpy(&programSource[0], AAsset_getBuffer(asset), length);
    AAsset_close(asset);

    jboolean iscopy = false;
    const char* url = env->GetStringUTFChars(fileUrl, &iscopy);
    path = std::string(url);
    img = cv::imread((path + "/OpenCL/01.png").c_str());
    env->ReleaseStringUTFChars(fileUrl, url);
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

    cv::Mat ImgGray;
    cv::cvtColor(img, ImgGray, CV_BGRA2GRAY);
    const int width = img.cols;
    const int height = img.rows;
    const int channel = img.channels();
    float theta = 45 *  M_PI / 180;
    auto dataSize = ImgGray.total();


    uchar *bufIn = (uchar*)malloc(dataSize);
    uchar *bufOut = (uchar*)malloc(dataSize);

    memcpy(bufIn, ImgGray.data, dataSize);



//
//     for (int i = 0; i < heightC; ++i)
//     {
//         for (int j = 0; j < widthC; ++j)
//         {
//             out << C[i * widthC + j];
//         }
//         out << std::endl;
//     }
//
//     return 0;
    /*  clock_t start, end;*/

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

    //declare the buffer and transimit it
    auto t1 = std::chrono::high_resolution_clock::now();


    cl_mem bufferIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY, dataSize, NULL, &ciErrNum);
    if (ciErrNum == CL_SUCCESS){
        printf("buffer A created successfully!\n");
    }
    ciErrNum = clEnqueueWriteBuffer(myqueue, bufferIn, CL_FALSE, 0, dataSize, (void *)bufIn, 0, NULL, NULL);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (ciErrNum == CL_SUCCESS){
        printf("bufferA has transformed successfully!\n");
    }



    cl_mem bufferOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, dataSize, NULL, &ciErrNum);
    ciErrNum = clEnqueueWriteBuffer(myqueue, bufferOut, CL_FALSE, 0, dataSize, (void *)bufOut, 0, NULL, NULL);
    if (ciErrNum == CL_SUCCESS){
        printf("bufferB has transformed successfully!\n");
    }


    const char*srcKernel = programSource.c_str();
    //compile and build kernel
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&srcKernel, NULL, &ciErrNum);
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



    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    //set the kernel arguments
    ciErrNum = clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&bufferOut);
    ciErrNum |= clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void *)&bufferIn);
    ciErrNum |= clSetKernelArg(mykernel, 2, sizeof(cl_int), (void *)&width);
    ciErrNum |= clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&height);
    ciErrNum |= clSetKernelArg(mykernel, 4, sizeof(cl_int), (void *)&channel);
    ciErrNum |= clSetKernelArg(mykernel, 5, sizeof(cl_float), (void *)&cos_theta);
    ciErrNum |= clSetKernelArg(mykernel, 6, sizeof(cl_float), (void *)&sin_theta);
    printf("set kernel arguments successfully!\n");
    if (ciErrNum != CL_SUCCESS){
        printf("set kernel arguments wrong\n");
    }

    //size_t localws[2] = {16,16};
    size_t globalws[2] = {static_cast<size_t >(width), static_cast<size_t>(height)};

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
    ciErrNum = clEnqueueReadBuffer(myqueue, bufferOut, CL_TRUE, 0, dataSize, (void *)bufOut, 0, NULL, NULL);



    cv::Mat test(ImgGray.rows, ImgGray.cols, ImgGray.type());
/*    cv::Mat test(img.rows, img.cols, img.type());*/

    for (int iy = 0; iy < height; iy++)
    {
        for (int ix = 0; ix < width; ix++)
        {
            int xpos = ((float)(ix - width / 2)) * cos_theta + ((float)(-iy + height / 2)) * sin_theta + width / 2;
            int ypos = ((float)(ix - width / 2)) * sin_theta + ((float)(iy - height / 2)) * cos_theta + height / 2;

            if ((xpos >= 0) && (xpos < width) && (ypos >= 0) && (ypos < height))
            {
                test.data[ypos * width + xpos] = ImgGray.data[iy * width + ix];
//                 test.data[(((int)ypos) * width + (int)xpos) * channel] = img.data[(iy * width + ix) * channel];
//                 test.data[(((int)ypos) * width + (int)xpos) * channel + 1] = img.data[(iy * width + ix) * channel + 1];
//                 test.data[(((int)ypos) * width + (int)xpos) * channel + 2] = img.data[(iy * width + ix) * channel + 2];
            }

        }
    }


    /*   cv::Mat detImg(img.rows, img.cols, img.type(), bufOut);*/
    cv::Mat detImg(ImgGray.rows, ImgGray.cols, ImgGray.type(), bufOut);
    cv::imwrite(path + "/OpenCL/detImg.png", detImg);
    clReleaseKernel(mykernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(myqueue);
    clReleaseMemObject(bufferIn);
    clReleaseMemObject(bufferOut);
    clReleaseContext(ctx);

    free(bufIn);
    free(bufOut);

//    std::string hello = ;
//    return env->NewStringUTF((during + "       " + during2 + result).c_str());
    return  env->NewStringUTF(std::to_string(dur).c_str());
}

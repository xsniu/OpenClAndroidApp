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
    img = cv::imread((path + "/OpenCl/4.jpg").c_str());
    env->ReleaseStringUTFChars(fileUrl, url);
    return 0;

}


void GetGaussianKernel(float **gaus, const int size, const double sigma)
{
    const double PI = 4.0*atan(1.0);
    int center = size / 2;
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            gaus[i][j] = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
            sum += gaus[i][j];
        }
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            gaus[i][j] /= sum;
        }
    }
    return;
}


extern "C"
JNIEXPORT jstring
JNICALL
Java_com_cloudream_myapplication_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    int filterSize = 9;
    float **gausFilter = new float *[filterSize];
    for (int i = 0; i < filterSize; i++)
    {
        gausFilter[i] = new float[filterSize];
    }

    GetGaussianKernel(gausFilter, 9, 10);

    float *filter = new float[filterSize * filterSize];
    float *tmp = filter;
    for (int i = 0; i < filterSize; ++i)
    {
        memcpy(tmp, gausFilter[i], filterSize * sizeof(float));
        tmp += filterSize;
    }

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

    cl_image_format format;
    format.image_channel_data_type = CL_UNSIGNED_INT8;
    format.image_channel_order = CL_R;

    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = {static_cast<size_t>(width), static_cast<size_t>(height), 1 };

    cl_mem bufferSrcImg = clCreateImage2D(ctx, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, &format, width, height, 0,  bufIn, &ciErrNum);
    if (ciErrNum == CL_SUCCESS)
    {
        printf("Create src image success");
    }



    auto m1 = std::chrono::high_resolution_clock::now();
    ciErrNum = clEnqueueWriteImage(myqueue, bufferSrcImg, CL_FALSE, origin, region, 0, 0, bufIn, 0, NULL, NULL);
    clFinish(myqueue);
    auto m2 = std::chrono::high_resolution_clock::now();
    auto dur3 = std::chrono::duration_cast<std::chrono::microseconds>(m2 - m1).count();

    cl_mem bufferDestImg = clCreateImage2D(ctx, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, &format, width, height, 0,  bufOut, &ciErrNum);
    if (ciErrNum == CL_SUCCESS)
    {
        printf("Create dest image success");
    }

    cl_mem bufferFilter = clCreateBuffer(ctx, 0, filterSize * filterSize * sizeof(float), NULL, NULL);



    ciErrNum = clEnqueueWriteBuffer(myqueue, bufferFilter, CL_FALSE, 0, filterSize * filterSize * sizeof(float), (void *)filter, 0, NULL, NULL);

    //declare the buffer and transimit it
//     auto t1 = std::chrono::high_resolution_clock::now();
//
//
//     auto t2 = std::chrono::high_resolution_clock::now();
//     auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//     std::cout << dur << std::endl;


    const char*srcKernel = programSource.c_str();
    //compile and build kernel
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&srcKernel, NULL, &ciErrNum);
    if (ciErrNum == CL_SUCCESS){
        printf("create program successfully!\n");
    }
    if (ciErrNum != CL_SUCCESS){
        printf("create program wrong\n");
    }

    ciErrNum = clBuildProgram(program, 0, nullptr, NULL, NULL, NULL);
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
    ciErrNum = clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&bufferDestImg);
    ciErrNum |= clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void *)&bufferSrcImg);
    ciErrNum |= clSetKernelArg(mykernel, 2, sizeof(cl_int), (void *)&width);
    ciErrNum |= clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&height);
    ciErrNum |= clSetKernelArg(mykernel, 4, sizeof(cl_mem), (void *)&bufferFilter);
    ciErrNum |= clSetKernelArg(mykernel, 5, sizeof(cl_int), (void *)&filterSize);

    printf("set kernel arguments successfully!\n");
    if (ciErrNum != CL_SUCCESS){
        printf("set kernel arguments wrong\n");
    }

    /*size_t localws[2] = {1,1};*/
    size_t globalws[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};

    //execute the kernel

    auto t1 = std::chrono::high_resolution_clock::now();
    ciErrNum = clEnqueueNDRangeKernel(myqueue, mykernel, 2, NULL, globalws, nullptr, 0, NULL, NULL);
    clFinish(myqueue);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();


    if (ciErrNum == CL_SUCCESS){
        printf("execute the kernel successfully!\n");
    }

    if (ciErrNum != CL_SUCCESS){
        printf("execute the kernel wrong\n");
    }
    //Read the output

    t1 = std::chrono::high_resolution_clock::now();
    ciErrNum = clEnqueueReadImage(myqueue, bufferDestImg, CL_TRUE, origin, region, 0, 0, (void*)bufOut, 0, nullptr, nullptr);
    t2 = std::chrono::high_resolution_clock::now();
    auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    cv::Mat test(ImgGray.rows, ImgGray.cols, ImgGray.type());
    int halfFilter = filterSize / 2;
    float sum = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = halfFilter; i < test.rows - halfFilter; ++i)
    {
        for (int j = halfFilter; j < test.cols - halfFilter; ++j)
        {
            sum = 0.0;
            for (int k = -halfFilter; k <= halfFilter; ++k)
            {
                for (int l = -halfFilter; l <= halfFilter; ++l)
                {
                    //sum += ImgGray.data[(i + k) * width + j + l] * gausFilter[k + halfFilter][l + halfFilter];
                    sum += ImgGray.data[(i + k) * width + j + l] * filter[(k + halfFilter) * filterSize + l + halfFilter];
                }
            }
            test.data[i * width + j] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();



    cv::imwrite(path + "/OpenCl/test.png", test);

// /*    cv::Mat test(img.rows, img.cols, img.type());*/
//     for (int iy = 0; iy < height; iy++)
//     {
//         for (int ix = 0; ix < width; ix++)
//         {
//             int xpos = ((float)(ix - width / 2)) * cos_theta + ((float)(-iy + height / 2)) * sin_theta + width / 2;
//             int ypos = ((float)(ix - width / 2)) * sin_theta + ((float)(iy - height / 2)) * cos_theta + height / 2;
//
//             if ((xpos >= 0) && (xpos < width) && (ypos >= 0) && (ypos < height))
//             {
//                 test.data[ypos * width + xpos] = ImgGray.data[iy * width + ix];
// //                 test.data[(((int)ypos) * width + (int)xpos) * channel] = img.data[(iy * width + ix) * channel];
// //                 test.data[(((int)ypos) * width + (int)xpos) * channel + 1] = img.data[(iy * width + ix) * channel + 1];
// //                 test.data[(((int)ypos) * width + (int)xpos) * channel + 2] = img.data[(iy * width + ix) * channel + 2];
//             }
//
//         }
//     }


    /*   cv::Mat detImg(img.rows, img.cols, img.type(), bufOut);*/
    cv::Mat detImg(ImgGray.rows, ImgGray.cols, ImgGray.type(), bufOut);

    cv::imwrite(path + "/OpenCl/det.png", detImg);

    clReleaseKernel(mykernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(myqueue);
    clReleaseMemObject(bufferDestImg);
    clReleaseMemObject(bufferSrcImg);
    clReleaseContext(ctx);

    free(bufIn);
    free(bufOut);

//    return  env->NewStringUTF("hello");
    return  env->NewStringUTF((std::to_string(dur1) + "  " + std::to_string(dur2) + "  " + std::to_string(dur3) +  "   " + std::to_string(dur)).c_str());
}

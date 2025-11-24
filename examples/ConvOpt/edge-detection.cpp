//====- edge-detection.cpp - Example of lexon-opt tool --------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements an edge detection example with linalg.conv_2d operation.
// The linalg.conv_2d operation will be compiled into an object file with the
// lexon-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include "../kernels.h"
#include <lexon/DIP/ImageContainer.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <time.h>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_conv_2d(Img<float, 2> *input, MemRef<float, 2> *kernel,
                          MemRef<float, 2> *output);
}

int main(int argc, char *argv[]) {
  cout << "Start processing..." << endl;
  cout << "-----------------------------------------" << endl;
  //-------------------------------------------------------------------------//
  // Lexon Conv2D
  //-------------------------------------------------------------------------//

  /// Evaluate Lexon reading input process.
  clock_t lexonReadStart;
  lexonReadStart = clock();
  // Read as grayscale image.
  Mat lexonInputMat = imread(argv[1], IMREAD_GRAYSCALE);
  Img<float, 2> lexonInputMemRef(lexonInputMat);
  clock_t lexonReadEnd;
  lexonReadEnd = clock();
  double lexonReadTime =
      (double)(lexonReadEnd - lexonReadStart) / CLOCKS_PER_SEC;
  cout << "[Lexon] Read input time: " << lexonReadTime << " s" << endl;

  /// Evaluate Lexon defining kernel process.
  clock_t lexonKernelStart;
  lexonKernelStart = clock();
  // Get the data, row, and column information of the kernel.
  float *kernelAlign = laplacianKernelAlign;
  int kernelRows = laplacianKernelRows;
  int kernelCols = laplacianKernelCols;
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  // Define the kernel MemRef object.
  MemRef<float, 2> kernelMemRef(kernelAlign, sizesKernel);
  clock_t lexonKernelEnd;
  lexonKernelEnd = clock();
  double lexonKernelTime =
      (double)(lexonKernelEnd - lexonKernelStart) / CLOCKS_PER_SEC;
  cout << "[Lexon] Define kernel time: " << lexonKernelTime << " s" << endl;

  /// Evaluate Lexon defining the output.
  clock_t lexonOutputStart;
  lexonOutputStart = clock();
  // Define the output.
  int outputRows = lexonInputMat.rows - kernelRows + 1;
  int outputCols = lexonInputMat.cols - kernelCols + 1;
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> outputMemRef(sizesOutput);
  clock_t lexonOutputEnd;
  lexonOutputEnd = clock();
  double lexonOutputTime =
      (double)(lexonOutputEnd - lexonOutputStart) / CLOCKS_PER_SEC;
  cout << "[Lexon] Read output time: " << lexonOutputTime << " s" << endl;

  /// Evaluate the Lexon Conv2D.
  clock_t lexonConvStart;
  lexonConvStart = clock();
  // Perform the Conv2D function.
  _mlir_ciface_conv_2d(&lexonInputMemRef, &kernelMemRef, &outputMemRef);
  clock_t lexonConvEnd;
  lexonConvEnd = clock();
  double lexonConv2DTime =
      (double)(lexonConvEnd - lexonConvStart) / CLOCKS_PER_SEC;
  cout << "[Lexon] Perform Conv2D time: " << lexonConv2DTime << " s" << endl;

  /// Evaluate OpenCV writing output to image.
  clock_t lexonWriteStart;
  lexonWriteStart = clock();
  // Define a cv::Mat with the output of the Conv2D.
  Mat outputImage(outputRows, outputCols, CV_32FC1, outputMemRef.getData());
  // Choose a PNG compression level
  vector<int> lexonCompressionParams;
  lexonCompressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  lexonCompressionParams.push_back(9);
  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("lexon-result.png", outputImage, lexonCompressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  clock_t lexonWriteEnd;
  lexonWriteEnd = clock();
  double lexonWriteTime =
      (double)(lexonWriteEnd - lexonWriteStart) / CLOCKS_PER_SEC;
  cout << "[Lexon] Write image time: " << lexonWriteTime << " s" << endl;

  if (result) {
    cout << "[Lexon] Read + Conv2D time: "
         << lexonReadTime + lexonKernelTime + lexonOutputTime + lexonConv2DTime
         << endl;
    cout << "[Lexon] Total time: "
         << lexonReadTime + lexonKernelTime + lexonOutputTime +
                lexonConv2DTime + lexonWriteTime
         << endl;
  } else
    cout << "ERROR: Can't save PNG file." << endl;
  cout << "-----------------------------------------" << endl;

  //-------------------------------------------------------------------------//
  // OpenCV Filter2D
  //-------------------------------------------------------------------------//

  /// Evaluate OpenCV reading input process.
  clock_t ocvReadStart;
  ocvReadStart = clock();
  // Perform the read function
  Mat ocvInputImageFilter2D = imread(argv[1], IMREAD_GRAYSCALE);
  clock_t ocvReadEnd;
  ocvReadEnd = clock();
  double ocvReadTime = (double)(ocvReadEnd - ocvReadStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Read input time: " << ocvReadTime << " s" << endl;

  /// Evaluate OpenCV defining kernel process.
  clock_t ocvKernelStart;
  ocvKernelStart = clock();
  // Get the data, row, and column information of the kernel.
  float *ocvKernelAlign = laplacianKernelAlign;
  int ocvKernelRows = laplacianKernelRows;
  int ocvKernelCols = laplacianKernelCols;
  // Define the kernel cv::Mat object.
  Mat ocvKernelFilter2D =
      Mat(ocvKernelRows, ocvKernelCols, CV_32FC1, ocvKernelAlign);
  clock_t ocvKernelEnd;
  ocvKernelEnd = clock();
  double ocvKernelTime =
      (double)(ocvKernelEnd - ocvKernelStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Define kernel time: " << ocvKernelTime << " s" << endl;

  /// Evaluate OpenCV defining the output.
  clock_t ocvOutputStart;
  ocvOutputStart = clock();
  // Define a cv::Mat as the output object.
  Mat ocvOutputFilter2D;
  clock_t ocvOutputEnd;
  ocvOutputEnd = clock();
  double ocvOutputTime =
      (double)(ocvOutputEnd - ocvOutputStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Define output time: " << ocvOutputTime << " s" << endl;

  /// Perform the OpenCV Filter2D.
  clock_t ocvFilter2DStart;
  ocvFilter2DStart = clock();
  // Perform the function with the input, output, and kernel cv::Mat object.
  filter2D(ocvInputImageFilter2D, ocvOutputFilter2D, CV_32FC1,
           ocvKernelFilter2D, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
  clock_t ocvFilter2DEnd;
  ocvFilter2DEnd = clock();
  double ocvFilter2DTime =
      (double)(ocvFilter2DEnd - ocvFilter2DStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Filter2D time: " << ocvFilter2DTime << " s" << endl;

  /// Write the OpenCV output to image.
  clock_t ocvWriteStart;
  ocvWriteStart = clock();
  // Choose a PNG compression level
  vector<int> ocvCompressionParams;
  ocvCompressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  ocvCompressionParams.push_back(9);
  // Write output to PNG.
  bool ocvResult = false;
  try {
    ocvResult =
        imwrite("opencv-result.png", ocvOutputFilter2D, ocvCompressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  clock_t ocvWriteEnd;
  ocvWriteEnd = clock();
  double ocvWriteTime = (double)(ocvWriteEnd - ocvWriteStart) / CLOCKS_PER_SEC;
  cout << "[OpenCV] Write image time: " << ocvWriteTime << " s" << endl;

  if (ocvResult) {
    cout << "[OpenCV] Read + Filter2D time: "
         << ocvReadTime + ocvKernelTime + ocvOutputTime + ocvFilter2DTime
         << endl;
    cout << "[OpenCV] Total time: "
         << ocvReadTime + ocvKernelTime + ocvOutputTime + ocvFilter2DTime +
                ocvWriteTime
         << endl;
  } else
    cout << "ERROR: Can't save PNG file." << endl;
  cout << "-----------------------------------------" << endl;
  return 0;
}

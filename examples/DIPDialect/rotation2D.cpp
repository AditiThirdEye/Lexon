//====- rotation2D.cpp - Example of lexon-opt tool ===========================//
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
// This file implements a 2D rotation example with dip.rotate_2d operation.
// The dip.rotate_2d operation will be compiled into an object file with the
// lexon-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include "lexon/DIP/imgcodecs/loadsave.h"
#include <lexon/Core/Container.h>
#include <lexon/DIP/DIP.h>
#include <lexon/DIP/ImageContainer.h>
#include <iostream>
#include <math.h>

using namespace std;

bool testImplementation(int argc, char *argv[]) {
  // Read as grayscale image.
  Img<float, 2> input = dip::imread<float, 2>(argv[1], dip::IMGRD_GRAYSCALE);

  MemRef<float, 2> output = dip::Rotate2D(&input, 45, dip::ANGLE_TYPE::DEGREE);

  // Define a Img with the output of Rotate2D.
  intptr_t sizes[2] = {output.getSizes()[0], output.getSizes()[1]};

  Img<float, 2> outputImageRotate2D(output.getData(),sizes);

  dip::imwrite(argv[2], outputImageRotate2D);

  return 1;
}

int main(int argc, char *argv[]) {
  testImplementation(argc, argv);

  return 0;
}
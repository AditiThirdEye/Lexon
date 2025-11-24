_add_test_executable(lexon-container-test
  ContainerTest.cpp
)

if(LEXON_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(LEXON_MLIR_ENABLE_DIP_LIB AND LEXON_ENABLE_OPENCV)
  set(DIP_LIBS ${JPEG_LIBRARY} ${PNG_LIBRARY})
  _add_test_executable(lexon-image-container-test
    ImageContainerTest.cpp
    LINK_LIBS
      ${OpenCV_LIBS}
      ${DIP_LIBS}
  )
endif()

if(LEXON_MLIR_ENABLE_DIP_LIB)
  set(NEW_DIP_LIBS "")
  if(LEXON_ENABLE_PNG)
    list(APPEND NEW_DIP_LIBS ${PNG_LIBRARIES})
    _add_test_executable(lexon-new-image-container-test-png
      NewImageContainerTestPng.cpp
      LINK_LIBS
        ${NEW_DIP_LIBS}
    )
  endif()
  _add_test_executable(lexon-new-image-container-test-bmp
    NewImageContainerTestBmp.cpp
    LINK_LIBS
      ${NEW_DIP_LIBS}
  )
endif()

_add_test_executable(lexon-audio-container-test
  AudioContainerTest.cpp
)

_add_test_executable(lexon-text-container-test
  TextContainerTest.cpp
)

cmake_minimum_required(VERSION 3.5)

project(test_onnx_ncnn)

set(CMAKE_CXX_STANDARD 11)

include(GNUInstallDirs)

find_package(OpenCV REQUIRED)
find_package(Vulkan REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${Vulkan_INCLUDE_DIRS}
                    d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/include/ncnn
                    d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/include/glslang)

add_executable(wuSNet_test wuSNet_test.cpp)
add_executable(test_onnx_ncnn test_onnx_ncnn.cpp)
add_executable(verify_pytorch_ncnn test_pytorch_to_ncnn/verify_pytorch_ncnn.cpp) # added by Holy 2108310810

target_link_libraries(wuSNet_test ${OpenCV_LIBS} 
                                  ${Vulkan_LIBRARIES}
                                  d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                  d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                  d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                  d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                  d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                  d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                  d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)

target_link_libraries(test_onnx_ncnn ${OpenCV_LIBS} 
                                     ${Vulkan_LIBRARIES}
                                     d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                     d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                     d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                     d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                     d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                     d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                     d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)

# added by Holy 2108310810
target_link_libraries(verify_pytorch_ncnn ${OpenCV_LIBS} 
                                          ${Vulkan_LIBRARIES}
                                          d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                          d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                          d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                          d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                          d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                          d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                          d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)
# end of addition 2108310810

install(TARGETS wuSNet_test 
                test_onnx_ncnn
                verify_pytorch_ncnn
        DESTINATION ${CMAKE_INSTALL_BINDIR})
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/run_wu.ps1 
                 ${CMAKE_CURRENT_SOURCE_DIR}/run_test.ps1 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_pytorch_to_ncnn/resnet18.bin 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_pytorch_to_ncnn/resnet18.param 
        DESTINATION ${CMAKE_INSTALL_BINDIR})
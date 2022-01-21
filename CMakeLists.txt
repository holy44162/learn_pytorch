cmake_minimum_required(VERSION 3.5)

project(test_onnx_ncnn)

set(CMAKE_CXX_STANDARD 11)

include(GNUInstallDirs)

find_package(OpenCV REQUIRED)
find_package(Vulkan REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${Vulkan_INCLUDE_DIRS}
                    d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/include/ncnn
                    d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/include/glslang
                    d:/backup/project/learn_pytorch/test_cutpaste)

add_executable(wuSNet_test wuSNet_test.cpp)
add_executable(test_onnx_ncnn test_onnx_ncnn.cpp)
add_executable(verify_pytorch_ncnn test_pytorch_to_ncnn/verify_pytorch_ncnn.cpp) # added by Holy 2108310810
add_executable(verify_pytorch_ncnn_holy test_pytorch_to_ncnn/verify_pytorch_ncnn_holy.cpp) # added by Holy 2108310810
add_executable(verify_pytorch_ncnn_shufflenet_v2 test_shufflenet_v2/_sixth_verify_pytorch_ncnn_shufflenet_v2.cpp) # added by Holy 2109021500
add_executable(eighth_onnx_ncnn test_shufflenet_v2/eighth_onnx_ncnn.cpp) # added by Holy 2109041002
add_executable(eighth_onnx_ncnn_one_class test_shufflenet_v2/eighth_onnx_ncnn_one_class.cpp) # added by Holy 2110131500
add_executable(eighth_onnx_ncnn_oc test_cutpaste/eighth_onnx_ncnn_oc.cpp
                                   test_cutpaste/mman.c) # added by Holy 2111181500
add_executable(demo_numpy_c test_cutpaste/demo_numpy_c.cpp) # added by Holy 2111190810

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

target_link_libraries(verify_pytorch_ncnn_holy ${OpenCV_LIBS} 
                                               ${Vulkan_LIBRARIES}
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)
# end of addition 2108310810

# added by Holy 2109021500
target_link_libraries(verify_pytorch_ncnn_shufflenet_v2 ${OpenCV_LIBS} 
                                               ${Vulkan_LIBRARIES}
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                               d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)
# end of addition 2109021500

# added by Holy 2109021500
target_link_libraries(eighth_onnx_ncnn ${OpenCV_LIBS} 
                                       ${Vulkan_LIBRARIES}
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)
# end of addition 2109021500

# added by Holy 2110131500
target_link_libraries(eighth_onnx_ncnn_one_class ${OpenCV_LIBS} 
                                                 ${Vulkan_LIBRARIES}
                                                 d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                                 d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                                 d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                                 d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                                 d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                                 d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                                 d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)
# end of addition 2110131500

# added by Holy 2111181500
target_link_libraries(eighth_onnx_ncnn_oc ${OpenCV_LIBS} 
                                       ${Vulkan_LIBRARIES}
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/ncnn.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/SPIRV.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/glslang.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/MachineIndependent.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/GenericCodeGen.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OSDependent.lib
                                       d:/backup/project/ncnn/ncnn-20210720-full-source/windows_vs2017/build/install/lib/OGLCompiler.lib)
# end of addition 2111181500

# added by Holy 2111191500
target_link_libraries(demo_numpy_c ${OpenCV_LIBS})
# end of addition 2111191500

install(TARGETS wuSNet_test 
                test_onnx_ncnn
                verify_pytorch_ncnn 
                verify_pytorch_ncnn_holy 
                verify_pytorch_ncnn_shufflenet_v2 
                eighth_onnx_ncnn 
                eighth_onnx_ncnn_one_class 
                eighth_onnx_ncnn_oc
                demo_numpy_c
        DESTINATION ${CMAKE_INSTALL_BINDIR})
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/run_wu.ps1 
                 ${CMAKE_CURRENT_SOURCE_DIR}/run_test.ps1 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_shufflenet_v2/run_eighth.ps1 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_shufflenet_v2/run_eighth_one_class.ps1 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_pytorch_to_ncnn/resnet18.bin 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_pytorch_to_ncnn/resnet18.param 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_shufflenet_v2/shufflenet_v2_x1_0.bin 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_shufflenet_v2/shufflenet_v2_x1_0.param 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_cutpaste/model_ft_shufflenet_v2_ckp_0.89_simplified.bin 
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_cutpaste/model_ft_shufflenet_v2_ckp_0.89_simplified.param
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_cutpaste/run_eighth_oc.ps1
        DESTINATION ${CMAKE_INSTALL_BINDIR})
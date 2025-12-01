cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

project(pyimagecuda_internal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;75;80;86;89")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")

python_add_library(pyimagecuda_internal MODULE ${SOURCES} WITH_SOABI)

target_include_directories(pyimagecuda_internal PRIVATE src)

target_link_libraries(pyimagecuda_internal PRIVATE CUDA::cudart_static Python::Module)

install(TARGETS pyimagecuda_internal DESTINATION pyimagecuda)
install(DIRECTORY pyimagecuda DESTINATION .)
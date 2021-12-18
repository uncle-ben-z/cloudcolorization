cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 17)

project(scene)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(happly)
include_directories(pugixml/src)
include_directories(cnpy)

link_directories(${CMAKE_SOURCE_DIR}/pugixml/build)
link_directories(${CMAKE_SOURCE_DIR}/cnpy/build)

pybind11_add_module(scene scene.cpp)

#add_executable (scene scene.cpp)
target_link_libraries (scene PUBLIC
  OpenMP::OpenMP_CXX 
  ${OpenCV_LIBS}
  pugixml
  cnpy
)

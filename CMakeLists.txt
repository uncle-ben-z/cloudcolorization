cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 17)

project(scene)

find_package(PCL 1.2 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(happly)
include_directories(pugixml/src)
include_directories(cnpy)

link_directories(${CMAKE_SOURCE_DIR}/pugixml/build)
link_directories(${CMAKE_SOURCE_DIR}/cnpy/build)

pybind11_add_module(scene scene.cpp)

#add_executable (scene scene.cpp)
target_link_libraries (scene PUBLIC
  OpenMP::OpenMP_CXX 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS} 
  pugixml
  cnpy
)

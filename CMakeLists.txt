cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 11)

project(colorize_point_cloud)

find_package(PCL 1.2 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(happly)
include_directories(pugixml/src)
include_directories(cnpy)

link_directories(${CMAKE_SOURCE_DIR}/pugixml/build)
link_directories(${CMAKE_SOURCE_DIR}/cnpy/build)

add_executable (colorize_point_cloud colorize_point_cloud.cpp)
target_link_libraries (colorize_point_cloud PUBLIC 
  OpenMP::OpenMP_CXX 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS} 
  pugixml
  cnpy
)

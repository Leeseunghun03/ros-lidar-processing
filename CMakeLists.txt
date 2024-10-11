cmake_minimum_required(VERSION 3.0.2)
project(lidar_processing)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_conversions
  pcl_ros
  sensor_msgs
  std_msgs
)

## Find PCL
find_package(PCL 1.8 REQUIRED)

## Find Armadillo
find_package(Armadillo REQUIRED)

## Find mlpack
# set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake/mlpack" ${CMAKE_PREFIX_PATH})
find_package(MLPACK REQUIRED)

## Find OpenMP
find_package(OpenMP REQUIRED)

## Declare a catkin package
catkin_package()

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${ARMADILLO_INCLUDE_DIRS}
  ${MLPACK_INCLUDE_DIRS}
)

## Specify libraries to link against
link_directories(${PCL_LIBRARY_DIRS})

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/lidar_processing_node.cpp)

## Link the libraries
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${MLPACK_LIBRARIES}
  OpenMP::OpenMP_CXX  # Add OpenMP support
)

## Add install directives (optional)
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

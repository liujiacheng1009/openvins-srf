cmake_minimum_required(VERSION 3.10)
project(mono_vio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -fno-omit-frame-pointer")

find_package(catkin QUIET COMPONENTS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs image_transport cv_bridge)

include(GNUInstallDirs)
set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/open_vins/")

message(STATUS "MANUALLY LINKING TO OV_MSCKF LIBRARY....")
file(GLOB_RECURSE OVMSCKF_LIBRARY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../msckf/src/*.cpp")
list(FILTER OVMSCKF_LIBRARY_SOURCES EXCLUDE REGEX ".*test_vio_simulation\\.cpp$")
list(FILTER OVMSCKF_LIBRARY_SOURCES EXCLUDE REGEX ".*test_sim_meas\\.cpp$")
list(FILTER OVMSCKF_LIBRARY_SOURCES EXCLUDE REGEX ".*test_sim_repeat\\.cpp$")
list(FILTER OVMSCKF_LIBRARY_SOURCES EXCLUDE REGEX ".*test_filters_srf\\.cpp$")
list(FILTER OVMSCKF_LIBRARY_SOURCES EXCLUDE REGEX ".*test_filters_srif\\.cpp$")
list(APPEND LIBRARY_SOURCES ${OVMSCKF_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../msckf/src/)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)

# Find glog - try find_package first, then fallback to pkg-config or manual find
find_package(glog QUIET)
if(NOT glog_FOUND)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(GLOG libglog)
    endif()
    if(GLOG_FOUND)
        set(glog_FOUND TRUE)
    else()
        # Manual find
        find_path(glog_INCLUDE_DIRS NAMES glog/logging.h PATHS /usr/include /usr/local/include)
        find_library(glog_LIBRARIES NAMES glog PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
        if(glog_INCLUDE_DIRS AND glog_LIBRARIES)
            set(glog_FOUND TRUE)
        endif()
    endif()
endif()

if(NOT glog_FOUND)
    message(FATAL_ERROR "glog library not found. Please install: sudo apt-get install libgoogle-glog-dev")
endif()

# Include our header files
include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${rosbag_INCLUDE_DIR}
        ../../dependency
)

# Add glog include directories if found manually
if(glog_INCLUDE_DIRS)
    include_directories(${glog_INCLUDE_DIRS})
elseif(GLOG_INCLUDE_DIRS)
    include_directories(${GLOG_INCLUDE_DIRS})
endif()

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${rosbag_LIBRARIES}
)

# Add glog library
if(TARGET glog::glog)
    list(APPEND thirdparty_libraries glog::glog)
elseif(glog_LIBRARIES)
    list(APPEND thirdparty_libraries ${glog_LIBRARIES})
elseif(GLOG_LIBRARIES)
    list(APPEND thirdparty_libraries ${GLOG_LIBRARIES})
else()
    list(APPEND thirdparty_libraries glog)
endif()

add_executable(mono_vio mono_vio.cpp bag_reader.cpp ${LIBRARY_SOURCES})
target_link_libraries(mono_vio core_lib init_lib msckf_lib ${thirdparty_libraries})
install(TARGETS mono_vio
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


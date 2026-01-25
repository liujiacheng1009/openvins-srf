cmake_minimum_required(VERSION 3.3)
project(ov_core)

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
# The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
        find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})

# check if we have our python libs files (will search for python3 then python2 installs)
# sudo apt-get install python-matplotlib python-numpy python-dev
# https://cmake.org/cmake/help/v3.10/module/FindPythonLibs.html
find_package(PythonLibs QUIET)
option(DISABLE_MATPLOTLIB "Disable or enable matplotlib plot scripts in ov_eval" OFF)
if (PYTHONLIBS_FOUND AND NOT DISABLE_MATPLOTLIB)
    add_definitions(-DHAVE_PYTHONLIBS=1)
    message(STATUS "PYTHON VERSION: " ${PYTHONLIBS_VERSION_STRING})
    message(STATUS "PYTHON INCLUDE: " ${PYTHON_INCLUDE_DIRS})
    message(STATUS "PYTHON LIBRARIES: " ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})
    list(APPEND thirdparty_libraries ${PYTHON_LIBRARIES})
endif ()

# We need c++14 for ROS2, thus just require it for everybody
# NOTE: To future self, hope this isn't an issue...
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")

# Find our ROS version!
# NOTE: Default to using the ROS1 package if both are in our enviroment
# NOTE: https://github.com/romainreignier/share_ros1_ros2_lib_demo

include(GNUInstallDirs)
set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/open_vins/")

# Include our header files
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

include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
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

##################################################
# Make the core library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackKLT.cpp
        src/track/TrackSIM.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureDatabase.cpp
        src/feat/FeatureInitializer.cpp
        src/utils/print.cpp
        src/utils/trajectory_logger.cpp
        src/visualization/visualization.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(core_lib STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(core_lib ${thirdparty_libraries})
target_include_directories(core_lib PUBLIC src/)
install(TARGETS core_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##################################################
# Make binary files!
##################################################
if(${VIO_BUILD_TEST} STREQUAL "ON")
        add_executable(test_webcam src/test_webcam.cpp)
        target_link_libraries(test_webcam core_lib ${thirdparty_libraries})
        install(TARGETS test_webcam
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

        add_executable(test_profile src/test_profile.cpp)
        target_link_libraries(test_profile core_lib ${thirdparty_libraries})
        install(TARGETS test_profile
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
endif()


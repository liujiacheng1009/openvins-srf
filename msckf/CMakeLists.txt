cmake_minimum_required(VERSION 3.3)
project(ov_msckf)

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
# The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
        find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(GTest REQUIRED)

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

message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})

# We need c++14 for ROS2, thus just require it for everybody
# NOTE: To future self, hope this isn't an issue...
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -fno-omit-frame-pointer")

include(GNUInstallDirs)
set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/open_vins/")



# Include our header files
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

# If we are not building with ROS then we need to manually link to its headers
# This isn't that elegant of a way, but this at least allows for building without ROS
# If we had a root cmake we could do this: https://stackoverflow.com/a/11217008/7718197
# But since we don't we need to basically build all the cpp / h files explicitly :(
message(STATUS "MANUALLY LINKING TO OV_CORE LIBRARY....")
file(GLOB_RECURSE OVCORE_LIBRARY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../core/src/*.cpp")
list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_profile\\.cpp$")
list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_webcam\\.cpp$")
list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_tracking\\.cpp$")
list(APPEND LIBRARY_SOURCES ${OVCORE_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core/src/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../core/src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

message(STATUS "MANUALLY LINKING TO OV_INIT LIBRARY....")
file(GLOB_RECURSE OVINIT_LIBRARY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../init/src/*.cpp")
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_init\\.cpp$")
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_mle\\.cpp$")
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_simulation\\.cpp$")
list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*Simulator\\.cpp$")
list(APPEND LIBRARY_SOURCES ${OVINIT_LIBRARY_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../init/src/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../init/src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##################################################
# Make the shared library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/sim/Simulator.cpp
        src/state/State.cpp
        src/state/FilterSRF.cpp
        src/state/FilterSRIF.cpp
        src/state/FilterEKF.cpp
        src/state/FilterWrapper.cpp
        src/state/FilterUtils.cpp
        src/state/Propagator.cpp
        src/core/VioManager.cpp
        src/core/VioManagerHelper.cpp
        src/update/UpdaterHelper.cpp
        src/update/UpdaterMSCKF.cpp
        src/update/UpdaterSLAM.cpp
        src/update/UpdaterZeroVelocity.cpp
)

file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(msckf_lib STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(msckf_lib init_lib core_lib ${thirdparty_libraries})
target_include_directories(msckf_lib PUBLIC src/)
install(TARGETS msckf_lib
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
        add_executable(test_vio_simulation src/test_vio_simulation.cpp)
        target_link_libraries(test_vio_simulation core_lib init_lib msckf_lib ${thirdparty_libraries})
        install(TARGETS test_vio_simulation
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

        add_executable(test_sim_meas src/test_sim_meas.cpp)
        target_link_libraries(test_sim_meas core_lib init_lib msckf_lib ${thirdparty_libraries})
        install(TARGETS test_sim_meas
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

        add_executable(test_sim_repeat src/test_sim_repeat.cpp)
        target_link_libraries(test_sim_repeat core_lib init_lib msckf_lib ${thirdparty_libraries})
        install(TARGETS test_sim_repeat
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

        add_executable(test_filters_srf src/test_filters_srf.cpp)
        target_link_libraries(test_filters_srf core_lib init_lib msckf_lib gtest gtest_main pthread ${thirdparty_libraries})
        install(TARGETS test_filters_srf
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

        add_executable(test_filters_srif src/test_filters_srif.cpp)
        target_link_libraries(test_filters_srif core_lib init_lib msckf_lib gtest gtest_main pthread ${thirdparty_libraries})
        install(TARGETS test_filters_srif
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
endif()
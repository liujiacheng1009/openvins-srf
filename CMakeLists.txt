cmake_minimum_required(VERSION 3.10)
project(ekf_vio)

include_directories(core/src)
include_directories(init/src)
include_directories(msckf/src)

option(VIO_BUILD_TEST "Build Unittest" OFF)
option(VIO_ENABLE_VISUALIZATION "Option of enabling visualization" OFF)
option(VIO_USE_FLOAT "VIO FILTER USE FLOAT" ON)

set(VIO_FILTER
    "EKF"
    CACHE STRING "Choose Filter from EKF SRF SRIF")
set_property(CACHE VIO_FILTER PROPERTY STRINGS EKF SRF SRIF)

if(${VIO_FILTER} MATCHES "SRF")
  add_definitions(-DVIO_USE_SRF)
  message(STATUS "VIO: USE SRF enabled")
  if(VIO_USE_FLOAT)
    add_definitions(-DVIO_USE_FLOAT)
    message(STATUS "VIO: USE FLOAT enabled")
  endif()
elseif(${VIO_FILTER} MATCHES "SRIF")
  add_definitions(-DVIO_USE_SRIF)
  message(STATUS "VIO: USE SRIF enabled")
  if(VIO_USE_FLOAT)
    add_definitions(-DVIO_USE_FLOAT)
    message(STATUS "VIO: USE FLOAT enabled")
  endif()
else()
  message(STATUS "VIO: USE EKF enabled")
  message(STATUS "VIO: USE DOUBLE enabled")
endif()

if(VIO_ENABLE_VISUALIZATION)
  add_definitions(-DVIO_ENABLE_VISUALIZATION)
  message(STATUS "VIO: Show debug image enabled")
endif()

add_subdirectory(application/vio_sim)

add_subdirectory(core)
add_subdirectory(init)
add_subdirectory(msckf)
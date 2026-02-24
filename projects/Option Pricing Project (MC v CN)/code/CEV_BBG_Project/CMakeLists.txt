cmake_minimum_required(VERSION 3.16)
project(cev_bbg_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(cev_bbg
  src/main.cpp
  src/cev.cpp
  src/csv.cpp
  src/bs.cpp
  src/pde_bs.cpp
  src/bbg.cpp
)

target_include_directories(cev_bbg PRIVATE
  include
  C:/msys64/ucrt64/include
)
cmake_minimum_required(VERSION 3.16)
project(MPI-DAOS VERSION 0.2.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_path(DAOS_INCLUDE_DIR NAMES daos.h HINTS /usr/include /opt/daos/include)
find_library(DAOS_LIB NAMES daos HINTS /usr/lib64 /opt/daos/lib64)

file(GLOB SRC_FILES "src/*.c")

add_library(MPI-DAOS SHARED ${SRC_FILES})

target_include_directories(MPI-DAOS PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DAOS_INCLUDE_DIR}
)

target_link_libraries(MPI-DAOS PRIVATE
    Threads::Threads
    ${DAOS_LIB}
    dl
)

target_compile_options(MPI-DAOS PRIVATE -Wall -Wextra -fPIC)

cmake_minimum_required(VERSION 3.16)
project(MPI-DAOS VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Locate MPI
find_package(MPI REQUIRED COMPONENTS C)
find_package(Threads REQUIRED)

# Locate DAOS headers and libraries
find_path(DAOS_INCLUDE_DIR NAMES daos.h HINTS /usr/include /opt/daos/include)
find_library(DAOS_LIB NAMES daos HINTS /usr/lib64 /opt/daos/lib64)
find_library(DAOS_COMMON_LIB NAMES daos_common HINTS /usr/lib64 /opt/daos/lib64)
find_library(GURT_LIB NAMES gurt HINTS /usr/lib64 /opt/daos/lib64)

if(NOT DAOS_INCLUDE_DIR OR NOT DAOS_LIB)
    message(FATAL_ERROR "DAOS SDK not found. Set -DDAOS_INCLUDE_DIR and -DDAOS_LIB.")
endif()

add_library(MPI-DAOS SHARED
    src/handle_table.c
    src/datatype_flatten.c
    src/mpi_file_ops.c
)

target_include_directories(MPI-DAOS
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${DAOS_INCLUDE_DIR}
)

target_link_libraries(MPI-DAOS
    PRIVATE
        MPI::MPI_C
        Threads::Threads
        ${DAOS_LIB}
        ${DAOS_COMMON_LIB}
        ${GURT_LIB}
        uuid
)

target_compile_options(MPI-DAOS PRIVATE -Wall -Wextra -fPIC)

install(TARGETS MPI-DAOS LIBRARY DESTINATION lib)

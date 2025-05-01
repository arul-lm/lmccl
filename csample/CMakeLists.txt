set(MSCCLPP_MAJOR "0")
set(MSCCLPP_MINOR "6")
set(MSCCLPP_PATCH "0")

set(MSCCLPP_SOVERSION ${MSCCLPP_MAJOR})
set(MSCCLPP_VERSION "${MSCCLPP_MAJOR}.${MSCCLPP_MINOR}.${MSCCLPP_PATCH}")

cmake_minimum_required(VERSION 3.25)
project(lmcclpp LANGUAGES C)
# Declare project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS include/nccl.h)

# add_library(lmccl_obj OBJECT)
# target_sources(lmccl_obj PRIVATE ${SOURCES})
# target_sources(lmccl_obj PUBLIC FILE_SET HEADERS FILES ${HEADERS})
# target_include_directories(lmccl_obj PRIVATE include SYSTEM)
# set_target_properties(lmccl_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

add_library(lmccl SHARED)
# target_link_libraries(lmccl PUBLIC lmccl_obj)
target_sources(lmccl PRIVATE ${SOURCES})
target_sources(lmccl PUBLIC FILE_SET HEADERS FILES ${HEADERS})
target_include_directories(lmccl PRIVATE include SYSTEM)
set_target_properties(lmccl PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

# add_library(lmccl_static STATIC)
# target_link_libraries(lmccl_static PUBLIC lmccl_obj)
# set_target_properties(lmccl_static PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

# install(TARGETS lmccl_obj
#   FILE_SET HEADERS DESTINATION ${INSTALL_PREFIX}/include)
message("CMSD" ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS lmccl
  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# install(TARGETS lmccl_static
#   ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)

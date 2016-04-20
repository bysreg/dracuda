cmake_minimum_required(VERSION 2.8)
project(dracuda)

message(STATUS "CMAKE_LIBRARY_PATH variable = ${CMAKE_LIBRARY_PATH}")
message(STATUS "CMAKE_PREFIX_PATH variable = ${CMAKE_PREFIX_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include(build/CMakeLists.txt)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${SDL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

add_subdirectory(application)
add_subdirectory(math)
add_subdirectory(p3)
add_subdirectory(scene)
add_subdirectory(tinyxml)

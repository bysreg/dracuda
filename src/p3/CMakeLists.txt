find_package(CUDA)

if(NOT WIN32)
	list(APPEND CUDA_NVCC_FLAGS "-O2")
else()
	list(APPEND CUDA_NVCC_FLAGS "")
endif()
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

CUDA_ADD_EXECUTABLE(p3 base64.cpp application.cpp camera_roam.cpp PoolScene.cpp imageio.cpp main.cpp raytracer_cuda.cu master.cpp master.hpp slave.hpp slave.cpp constants.cpp load_balancer.cpp)

target_link_libraries(p3 math ${SDL_LIBRARY}
                      ${PNG_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
                      ${GLEW_LIBRARIES}
                      ${Boost_LIBRARIES})

if(APPLE)
    target_link_libraries(p3)
endif()

install(TARGETS p3 DESTINATION ${PROJECT_SOURCE_DIR}/..)

find_package(CUDA)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;")
CUDA_ADD_EXECUTABLE(p3 main.cpp raytracer.cpp photon.cpp neighbor.cpp photonmap.cpp util.cpp randomgeo.cpp raytracer_cuda.cu)
target_link_libraries(p3 application math scene tinyxml ${SDL_LIBRARY}
                      ${PNG_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
                      ${GLEW_LIBRARIES})

if(APPLE)
    target_link_libraries(p3)
endif()

install(TARGETS p3 DESTINATION ${PROJECT_SOURCE_DIR}/..)

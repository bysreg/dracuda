find_package(CUDA)

if(NOT WIN32)
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;")
else()
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11;")
endif()


CUDA_ADD_EXECUTABLE(p3 main.cpp raytracer.cpp raytracer.hpp photon.cpp photon.hpp neighbor.cpp neighbor.hpp photonmap.cpp photonmap.hpp util.cpp util.hpp randomgeo.cpp randomgeo.hpp collisions.cpp collisions.hpp modelbody.cpp modelbody.hpp planebody.cpp planebody.hpp spherebody.cpp spherebody.hpp trianglebody.cpp trianglebody.hpp body.hpp physics.cpp physics.hpp spring.cpp spring.hpp raytracer_cuda.cu)

target_link_libraries(p3 application math scene tinyxml ${SDL_LIBRARY}
                      ${PNG_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
                      ${GLEW_LIBRARIES})

if(APPLE)
    target_link_libraries(p3)
endif()

install(TARGETS p3 DESTINATION ${PROJECT_SOURCE_DIR}/..)

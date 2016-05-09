ADD_EXECUTABLE(p3 base64.cpp application.cpp camera_roam.cpp PoolScene.cpp imageio.cpp main.cpp master.cpp master.hpp slave.hpp slave.cpp constants.cpp load_balancer.cpp raytracer_simd.cpp glew.cpp)

target_link_libraries(p3 math ${SDL_LIBRARY}
                      ${PNG_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
                      ${Boost_LIBRARIES})

if(APPLE)
    target_link_libraries(p3)
endif()

install(TARGETS p3 DESTINATION ${PROJECT_SOURCE_DIR}/..)

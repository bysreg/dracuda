add_executable(p3 main.cpp raytracer.cpp raytracer.hpp photon.cpp photon.hpp neighbor.cpp neighbor.hpp photonmap.cpp photonmap.hpp util.cpp util.hpp randomgeo.cpp randomgeo.hpp)
target_link_libraries(p3 application math scene tinyxml ${SDL_LIBRARY}
                      ${PNG_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
                      ${GLEW_LIBRARIES})

if(APPLE)
    target_link_libraries(p3)
endif()

install(TARGETS p3 DESTINATION ${PROJECT_SOURCE_DIR}/..)

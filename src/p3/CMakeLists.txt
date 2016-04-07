add_executable(p3 main.cpp raytracer.cpp raytracer.hpp photon.cpp photon.hpp neighbor.cpp neighbor.hpp photonmap.cpp photonmap.hpp util.cpp util.hpp randomgeo.cpp randomgeo.hpp collisions.cpp collisions.hpp modelbody.cpp modelbody.hpp planebody.cpp planebody.hpp spherebody.cpp spherebody.hpp trianglebody.cpp trianglebody.hpp body.hpp)

target_link_libraries(p3 application math scene tinyxml ${SDL_LIBRARY}
                      ${PNG_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
                      ${GLEW_LIBRARIES})

if(APPLE)
    target_link_libraries(p3)
endif()

install(TARGETS p3 DESTINATION ${PROJECT_SOURCE_DIR}/..)

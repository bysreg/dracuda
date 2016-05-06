#pragma once

#include "application.hpp"
#include "camera_roam.hpp"
#include "options.hpp"

#include <SDL.h>

class RaytracerApplication : public Application
{
public:
    RaytracerApplication( const Options& opt )
        : options( opt ), cur_frame_number(0) {}

    virtual ~RaytracerApplication() {
		if (buffer)
			free( buffer );
	}

    virtual bool initialize();
    virtual void destroy();
    virtual void update( float );
    virtual void render();
    virtual void handle_event( const SDL_Event& event );
	float time;

	void do_gpu_raytracing();

    Options options;
    CameraRoamControl camera_control;
    unsigned int cur_frame_number;
    unsigned char* buffer = 0;
};

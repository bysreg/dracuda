#include "application.hpp"
#include "camera_roam.hpp"
#include <typeinfo>
#include "opengl.hpp"
#include "cudaScene.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "raytracer_cuda.hpp"
#include "cycleTimer.h"
#include "constants.hpp"
#include "PoolScene.hpp"

#include "master.hpp"
#include "slave.hpp"
#include "base64.h"

#include <SDL.h>

#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <string>

using namespace std;

unsigned char *cudaBuffer;
unsigned char* buffer = 0;

PoolScene poolScene;
CudaScene cudaScene;

Master* master;
Slave* slave;

void on_master_receive_message(const Message& message);
void on_slave_receive_message(const Message& message);

#define KEY_RAYTRACE_GPU SDLK_g

struct Options
{
	bool master = false;
	bool slave = false;
	std::string host; // host to connect from slave
};

class RaytracerApplication : public Application
{
public:
    RaytracerApplication( const Options& opt )
        : options( opt ){}

    virtual ~RaytracerApplication() {
		if (buffer)
			free( buffer );
	}

    virtual bool initialize();
    virtual void destroy();
    virtual void update( float );
    virtual void render();
    virtual void handle_event( const SDL_Event& event );
	float time = 0;

	void do_gpu_raytracing();

    Options options;
    CameraRoamControl camera_control;

};

int LoadEnvmap(cudaArray **array, const char *filename) {
	int width, height;
	float endian;
	FILE *file = fopen(filename, "r");
	char tmp[10];
	fscanf(file, "%s\n", tmp);
	fscanf(file, "%d %d\n", &width, &height);
	fscanf(file, "%f\n", &endian);
	int size = width * height * sizeof(float4) ;
	printf("HW: %d %d\n", width, height);
	float *data = (float *)malloc(size);
	fread(data, sizeof(float), height * width * 3, file);
	for (int i = height * width - 1; i >= 0; i--) {
		data[4 * i + 3] = 1.000;
		data[4 * i + 2] = data[3 * i + 2];
		data[4 * i + 1] = data[3 * i + 1];
		data[4 * i + 0] = data[3 * i + 0];
	}
	fclose(file);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat );
	gpuErrchk(cudaMallocArray(array, &channelDesc, width, height));
	gpuErrchk(cudaMemcpyToArray(*array, 0, 0, data, size, cudaMemcpyHostToDevice));
	free(data);
	return 0;
}


bool RaytracerApplication::initialize()
{
	poolScene.initialize();
	poolScene.camera.position = Vector3(0, 25, 0);
	poolScene.camera.orientation = Quaternion (0.717, -0.717, 0, 0);
	camera_control.camera = &poolScene.camera;
	if (!buffer) {
		buffer = new unsigned char [WIDTH * HEIGHT * 4];
	}

	cudaScene.fov = 0.785;
	cudaScene.aspect = (WIDTH + 0.0) / (HEIGHT + 0.0);
	cudaScene.near_clip = 0.01;
	cudaScene.far_clip = 200.0;

	cudaArray *cu_2darray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat );
	LoadEnvmap(&cu_2darray, "images/stpeters_probe.pfm");
	bindEnvmap(cu_2darray, channelDesc);

	// CUDA part
	gpuErrchk(cudaMalloc((void **)&cudaBuffer, 4 * HEIGHT * WIDTH));
	cudaInitialize();
	std::cout << "Cuda initialized" << std::endl;
	if(options.master) {
		master = &Master::start();
		master->set_on_message_received(on_master_receive_message);
	}else if(options.slave) {
		slave = &Slave::start(options.host);
		slave->set_on_message_received(on_slave_receive_message);
		slave->run();
	}	

    return true;
}

void RaytracerApplication::destroy()
{
}

Quaternion FromToRotation(Vector3 u, Vector3 v)
{
	Vector3 w = cross(u, v);
	Quaternion q(1.f + dot(u, v), w.x, w.y, w.z);
	return normalize(q);
}


void RaytracerApplication::update( float delta_time )
{
	camera_control.update(delta_time);
	if (options.master) {
		time += delta_time;
		poolScene.update(delta_time);
		poolScene.toCudaScene(cudaScene);
		master->send_all(cudaScene);
	} else if (!options.slave) {
		time += delta_time;
		poolScene.update(delta_time);
		poolScene.toCudaScene(cudaScene);
		cudaRayTrace(&cudaScene, cudaBuffer);
		gpuErrchk(cudaMemcpy(buffer, cudaBuffer, 4 * WIDTH * HEIGHT, cudaMemcpyDeviceToHost));
	}
}

void RaytracerApplication::render()
{
    glViewport( 0, 0, WIDTH, HEIGHT );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // reset matrices
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

	if (!buffer) {
		buffer = new unsigned char [WIDTH * HEIGHT * 4];
	}
	glColor4d( 1.0, 1.0, 1.0, 1.0 );
	glRasterPos2f( -1.0f, -1.0f );
	if (!options.slave)
		glDrawPixels( WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, &buffer[0] );
}

void RaytracerApplication::handle_event( const SDL_Event& event )
{
	camera_control.handle_event( event );

    switch ( event.type )
    {
    case SDL_KEYDOWN:
        switch ( event.key.keysym.sym )
        {
		case KEY_RAYTRACE_GPU:
			//do_gpu_raytracing();
			break;
        default:
            break;
        }
    default:
        break;
    }
}

void RaytracerApplication::do_gpu_raytracing()
{
}

static bool parse_args( Options* opt, int argc, char* argv[] )
{
    for (int i = 2; i < argc; i++)
    {
    	if(strcmp(argv[i] + 1, "master") == 0) {    		
    		opt->master = true;
    		continue;
    	}
    	if(strcmp(argv[i] + 1, "slave") == 0) {
    		// slave needs host address
    		if(i+1 > argc-1) {
    			std::cout<<"slave needs host address"<<std::endl;
    			return false;
    		}

    		opt->slave = true;
    		opt->host = argv[i + 1]; // we assume the next parameter is the master's host for the slave to connect to
    		i++;
    		continue;
    	}
    }
    return true;
}

using namespace std;

void on_master_receive_message(const Message& message)
{
	// we receive the image from slave
	// for now just print it

//	printf("BODY LENG %d\n", message.body_length());
	std::memcpy(buffer, message.body(), message.body_length());
	/*
	std::cout<<"receive : ";
	std::cout.write(message.body(), message.body_length());
	std::cout << "\n";
	*/
}

void on_slave_receive_message(const Message& message) 
{
	CudaScene cudaSceneCopy;
	std::memcpy(&cudaSceneCopy, message.body(), message.body_length());
	cudaRayTrace(&cudaSceneCopy, cudaBuffer);
	gpuErrchk(cudaMemcpy(buffer, cudaBuffer, 4 * WIDTH * HEIGHT, cudaMemcpyDeviceToHost));

	//std::string encoded_str = base64_encode(buffer, 4 * WIDTH * HEIGHT);
	slave->send(buffer, WIDTH * HEIGHT * 4);
	//slave->send(encoded_str);
}

int main( int argc, char* argv[] )
{
    Options opt;
	int ret = 0;

    if ( !parse_args( &opt, argc, argv ) ) {
        return 1;
    }

    RaytracerApplication app( opt );
	cout << "master:slave => " << opt.master << ":" << opt.slave << endl;

	float fps = 20.0;
	const char* title = "DRACUDA";

	ret = Application::start_application(&app, WIDTH, HEIGHT, fps, title);

	return ret;
}

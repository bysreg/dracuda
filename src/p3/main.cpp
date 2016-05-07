#include "application.hpp"
#include "camera_roam.hpp"
#include <typeinfo>
#include "opengl.hpp"
#include "cudaScene.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "raytracer_cuda.hpp"
#include "raytracer_single.hpp"
#include "raytracer_simd.hpp"
#include "cycleTimer.h"
#include "constants.hpp"
#include "PoolScene.hpp"

#include "master.hpp"
#include "slave.hpp"
#include "slave_info.hpp"
#include "base64.h"
#include "load_balancer.hpp"
#include "raytracer_application.hpp"
#include "options.hpp"

#include <SDL.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <string>
#include <climits>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;


PoolScene poolScene;
CudaScene cudaScene;

static Master* master;
static Slave* slave;
static bool paused;
// offset for slave buffer
// the first double is used to store the rendering latency
static const int slave_buffer_img_offset = sizeof(double);

// master's related variable
static int buffer_frame_height = 0;
static bool send_scene_status = false; // can we send scene data to slave?
static SlaveInfo slaves_info[MAX_SLAVE] = {0}; // zero initialize array
static double slaves_weight[MAX_SLAVE] = {0}; // zero initialize array
static bool app_started = false;
static RaytracerApplication* s_app = nullptr;

void on_master_connection_started(Connection& conn);
void on_master_receive_message(int conn_idx, const Message& message);
void on_slave_receive_message(const Message& message);

#define KEY_RAYTRACE_GPU SDLK_g

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
	time = 0;
	poolScene.initialize();
	poolScene.camera.position = Vector3(0, 25, 0);
	poolScene.camera.orientation = Quaternion (0.717, -0.717, 0, 0);
	camera_control.camera = &poolScene.camera;
	if (!buffer) {
		if(options.slave) {
			buffer = new unsigned char [WIDTH * HEIGHT * 4 + slave_buffer_img_offset];
		}else{
			buffer = new unsigned char [WIDTH * HEIGHT * 4];
		}

		// master needs additional buffer for double buffering
		if(options.master) {
			back_buffer = new unsigned char [WIDTH * HEIGHT * 4];
		}		
	}

	if (!options.master && !options.slave) {
		cudaScene.y0 = 0;
		cudaScene.render_height = HEIGHT;
	}

	// CUDA part
	cudaInitialize();
	simdInitialize();
	std::cout << "Cuda initialized" << std::endl;
	if(options.master) {
		// initialize master

		// master's read buffer need to be able to accomodate
		// image that is being sent from the slave
		Master::read_msg_max_length = WIDTH * HEIGHT * 4 + 100;
		master = &Master::start();
		master->set_on_message_received(on_master_receive_message);
		master->set_on_connection_started(on_master_connection_started);
		send_scene_status = true;
	}else if(options.slave) {
		// initialize slave

		// slave only needs to read scene's data from master
		Slave::read_msg_max_length = sizeof(cudaScene);
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

int distribute(double w, int total, double& amortized)
{
	double real, natural;
	real = w * total + amortized;
	natural = std::floor(real);
	amortized = real - natural;

	return natural;
}

void assign_work()
{
	int n = master->get_connections_count();

	if(n == 0 || !send_scene_status)
		return;
	
	send_scene_status = false;
	// std::cout<<std::endl;

	LoadBalancer::calc(s_app, slaves_info, slaves_weight, n);
	int sum_height = 0;	
	double amortized = 0;
	for(int i=0;i<n-1;i++)
	{		
		// always splitting equally for now
		slaves_info[i].render_height = distribute(slaves_weight[i], HEIGHT, amortized);
		sum_height += slaves_info[i].render_height;
	}
	slaves_info[n-1].render_height = HEIGHT - sum_height;

	int cur_y0 = 0;
	for(int i=0;i<n;i++)
	{		
		slaves_info[i].y0 = cur_y0;
		cudaScene.y0 = slaves_info[i].y0;
		cudaScene.render_height = slaves_info[i].render_height;
		slaves_info[i].send_time = CycleTimer::currentSeconds();
		
		master->send(i, cudaScene);

		cur_y0 += slaves_info[i].render_height;
	}
}

void RaytracerApplication::update( float delta_time )
{
	// don't update until we are ready to start 
	if (options.slave || options.master) {
		if(!app_started)
			return;
	}

	cur_frame_number = (cur_frame_number + 1) % UINT_MAX;
	camera_control.update(delta_time);
	if (options.master) {
		time += delta_time;
		if (!paused) {
			poolScene.update(delta_time);
		}
		poolScene.toCudaScene(cudaScene);
		assign_work();
	} else if (!options.slave) {
		// not master and not slave
		time += delta_time;
		poolScene.update(delta_time);
		poolScene.toCudaScene(cudaScene);
		//simdRayTrace(&cudaScene, buffer);
		//singleRayTrace(&cudaScene, buffer);
		cudaRayTrace(&cudaScene, buffer);
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

	glColor4d( 1.0, 1.0, 1.0, 1.0 );
	glRasterPos2f( -1.0f, -1.0f );
	if (!options.slave) {
		glDrawPixels( WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, &buffer[0] );
	}
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
			poolScene.balls[0].velocity += Vector3(0.0, 0.0, -5.0);
			break;
		case SDLK_h:
			paused = !paused;
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

void RaytracerApplication::swap_buffer()
{
	unsigned char* temp = buffer;
	buffer = back_buffer;
	back_buffer = temp;
}

static bool parse_args( Options* opt, int argc, char* argv[] )
{
    for (int i = 1; i < argc; i++)
    {
    	if(strcmp(argv[i] + 1, "master") == 0) {    		
    		opt->master = true;

    		if(i+1 > argc-1) {
    			std::cout<<"master needs number of minimum connected slaves"<<std::endl;
    			return false;
    		}
    		
    		// we assume the next parameter for master
    		// is the minimum slave required to begin application
    		opt->min_slave_to_start = std::atoi(argv[i + 1]);    		

    		continue;
    	}
    	else if(strcmp(argv[i] + 1, "slave") == 0) {
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

void on_master_connection_started(Connection& conn)
{
	int n = master->get_connections_count();

	if(n >= s_app->options.min_slave_to_start) {
		// minimum slave count is reached, now
		// we can start running the application
		app_started = true;		
	}
}

void on_master_receive_message(int conn_idx, const Message& message)
{
	// std::cout<<"start processing message from slave"<<std::endl;

	SlaveInfo& si = slaves_info[conn_idx];
	si.messages_received++;	

	// update the slave's response time data
	si.response_duration = CycleTimer::currentSeconds() - si.send_time;

	// grab the rendering time from the front of the message
	std::memcpy(&si.rendering_latency, message.body(), sizeof(double));

	// network latency is response_duration - rendering_latency
	si.network_latency = si.response_duration - si.rendering_latency;
	si.sum_network_latency += si.network_latency;

	// calc the rendering factor
	si.rendering_factor = si.rendering_latency / si.render_height;
	si.sum_rendering_factor += si.rendering_factor;

	// std::cout<<"receive msg " 
	// 	<< conn_idx << " "
	// 	<< ((message.body_length() - slave_buffer_img_offset) / 4 / WIDTH) << " "
	// 	<<"dur:"<< si.response_duration << " " 
	// 	<<"net:"<< si.network_latency  << " " 
	// 	<<"renlat:"<< si.rendering_latency << " " 
	// 	<<"anet:"<< si.get_avg_network_latency() << " "
	// 	<<"arenfac:"<< si.get_avg_rendering_factor() << " " 
	// 	<<std::endl;

	// we receive the image from slave-i	
	int byte_offset = si.y0 * WIDTH * 4;
	std::memcpy(s_app->back_buffer + byte_offset, message.body() + slave_buffer_img_offset, message.body_length() - sizeof(double));

	// we could only send the next scene data to slave
	// only if we have all the image pieces from the slaves
	int piece_height = ((message.body_length() / 4) / WIDTH);
	buffer_frame_height += piece_height;

	if(buffer_frame_height >= HEIGHT) 
	{
		buffer_frame_height = 0;
		send_scene_status = true;
		s_app->swap_buffer();
	}

	// std::cout<<"finish"<<std::endl;
}

void on_slave_receive_message(const Message& message) 
{
	CudaScene cudaSceneCopy;

	// calculate the rendering start time
	double rendering_start = CycleTimer::currentSeconds();

	std::memcpy(&cudaSceneCopy, message.body(), message.body_length());
	cudaRayTrace(&cudaSceneCopy, s_app->buffer + slave_buffer_img_offset);

	//int offset = cudaSceneCopy.y0 * WIDTH * 4;

	// calculate the rendering latency
	double rendering_latency = CycleTimer::currentSeconds() 
		- rendering_start;

	// put the rendering time in front of the buffer
	std::memcpy(s_app->buffer, &rendering_latency, sizeof(rendering_latency));

	//std::string encoded_str = base64_encode(buffer, 4 * WIDTH * HEIGHT);
	int height = cudaSceneCopy.render_height;
	slave->send(s_app->buffer, WIDTH * (height) * 4 + sizeof(rendering_latency));	
	//slave->send(encoded_str);
}

int main( int argc, char* argv[] )
{
	Options opt;
	opt.master = false;
	opt.slave = false;
	int ret = 0;

	if ( !parse_args( &opt, argc, argv ) ) {
	    return 1;
	}

	RaytracerApplication app( opt );
	s_app = &app;
	cout << "master:slave => " << opt.master << ":" << opt.slave << endl;

	float fps = 20.0;
	const char* title = "DRACUDA";

	if(opt.master) {
		title = "DRACUDA - Master";
	}else if(opt.slave) {
		title = "DRACUDA - Slave";
	}
	bool show_window = !opt.slave;

	ret = Application::start_application(&app, WIDTH, HEIGHT, fps, title, show_window);

	return ret;
}

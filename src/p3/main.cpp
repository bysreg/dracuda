#include "application/application.hpp"
#include "application/camera_roam.hpp"
#include "application/imageio.hpp"
#include "application/scene_loader.hpp"
#include "application/opengl.hpp"
#include "scene/scene.hpp"
#include "scene/sphere.hpp"
#include "scene/triangle.hpp"
#include "p3/raytracer.hpp"
#include <typeinfo>
#include "scene/model.hpp"
#include "cudaScene.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "raytracer_cuda.hpp"
#include "cycleTimer.h"
#include "constants.hpp"

#include "master.hpp"
#include "slave.hpp"

#include <SDL.h>

#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <string>

cudaScene cscene;
cudaScene cscene_host;
using namespace std;

unsigned char *cimg;

namespace _462 {

#define DEFAULT_WIDTH 800
#define DEFAULT_HEIGHT 600
#define dwidth 800
#define dheight 600

#define BUFFER_SIZE(w,h) ( (size_t) ( 4 * (w) * (h) ) )
#define KEY_RAYTRACE_GPU SDLK_g

struct Ball {
	Vector3 position;
	Quaternion orientation;
	Vector3 acceleration;
	Vector3 velocity;
};
Ball balls[SPHERES];

static const size_t NUM_GL_LIGHTS = 8;
struct Options
{
    bool open_window;
    const char* input_filename;
    const char* output_filename;
    int width, height;
    int num_samples;
	bool master = false;
	bool slave = false;

	std::string host; // host to connect from slave
};

class RaytracerApplication : public Application
{
public:
    RaytracerApplication( const Options& opt )
        : options( opt ), buffer( 0 ), buf_width( 0 ),
		buf_height(0), gpu_raytracing(false) {}

    virtual ~RaytracerApplication() {
		if (buffer)
			free( buffer );
	}

    virtual bool initialize();
    virtual void destroy();
    virtual void update( real_t );
    virtual void render();
    virtual void handle_event( const SDL_Event& event );
	float time = 0;

    // flips raytracing, does any necessary initialization
	void do_gpu_raytracing();

    Scene scene;
    Options options;
    CameraRoamControl camera_control;
    // the image buffer for raytracing
    unsigned char* buffer = 0;
    // width and height of the buffer
    int buf_width, buf_height;
	bool gpu_raytracing;
};

bool RaytracerApplication::initialize()
{
    // copy camera into camera control so it can be moved via mouse
    camera_control.camera = scene.camera;
    bool load_gl = options.open_window;
	gpuErrchk(cudaMalloc((void **)&cscene.data, sizeof(float) * 7 * SPHERES));

	// Mirrored host mem
	cscene_host.data = (float *)malloc(sizeof(float) * 7 * SPHERES);
	cscene_host.position = cscene_host.data + 4 * SPHERES;
	cscene_host.rotation = cscene_host.data;
	

	for (size_t i = 0; i < SPHERES; i++) {
		Geometry *g = scene.get_geometries()[i];
		balls[i].position = g->position;
		balls[i].orientation = g->orientation;
	}
	balls[2].velocity = Vector3(0.9, 0, -0.80);

	scene.post_initialize();

	gpuErrchk(cudaMemcpy(cscene.data, cscene_host.data, sizeof(float) * 7 * SPHERES, cudaMemcpyHostToDevice));
	
	scene.camera.position.to_array(cscene.cam_position);
	scene.camera.orientation.to_array(cscene.cam_orientation);
	cscene.fov = scene.camera.fov;
	cscene.aspect = (dwidth + 0.0) / (dheight + 0.0);
	cscene.near_clip = scene.camera.near_clip;
	cscene.far_clip = scene.camera.far_clip;

	EnvMap &envmap = scene.envmap;
	if (envmap.enabled) {
		envmap.initialize();
	int num_faces  = 6;
	int width = envmap.posx.width;
	int height = envmap.posx.height;
	std::cout << "Width : " << width << std::endl;

	int size = num_faces * width * height * sizeof(uchar4) ;
	int face_size = width * height * sizeof (uchar4);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned );
	cudaArray *cu_3darray;
	cudaMalloc3DArray(&cu_3darray, &channelDesc, make_cudaExtent(width, height, num_faces), cudaArrayCubemap);
	unsigned char *envmap_array = new unsigned char [size];
	memcpy(envmap_array, envmap.posx.data, face_size);
	memcpy(envmap_array + face_size, envmap.negx.data, face_size);
	memcpy(envmap_array + face_size * 2, envmap.posy.data, face_size);
	memcpy(envmap_array + face_size * 3, envmap.negy.data, face_size);
	memcpy(envmap_array + face_size * 4, envmap.posz.data, face_size);
	memcpy(envmap_array + face_size * 5, envmap.negz.data, face_size);

	cudaMemcpyToArray(cu_3darray, 0, 0, envmap_array, size, cudaMemcpyHostToDevice);
	bindEnvmap(cu_3darray, channelDesc);
	}


	// CUDA part
	gpuErrchk(cudaMalloc((void **)&cimg, 4 * dheight * dwidth));
	cudaInitialize();
	std::cout << "Cuda initialized" << std::endl;
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

Vector3 velocity_acc[SPHERES];
int times[SPHERES];

void RaytracerApplication::update( real_t delta_time )
{
	time += delta_time;
	// Camera
	float time_scale = 0.5;
	Vector3 pos(8 * sin(time * time_scale), 4, 8 * cos(time * time_scale));
	Vector3 dir = -normalize(pos);
	Vector3 look(0, 0, -1);
	Vector3 up(0, 1, 0);
	Vector3 v = dir + up * -dot(up, dir);
	Quaternion q = FromToRotation(look, v);
	Quaternion ret = FromToRotation(v, dir) * q;
	pos = Vector3(0, 20, 0);
	ret = Quaternion(-0.707, 0.707, 0, 0);
	pos.to_array(cscene.cam_position);
	ret.to_array(cscene.cam_orientation);
	for (int i = 0; i < SPHERES; i++) {
		velocity_acc[i] = Vector3(0, 0, 0);
		times[i] = 0;
		printf("Vel %d: %f\n", i, length(balls[i].velocity));
	}

	// Collision between spheres
	for (int i = 0; i < SPHERES; i++) {
		for (int j = i + 1; j < SPHERES; j++) {
			Vector3 dist = -balls[i].position + balls[j].position;
			Vector3 vel1 = balls[i].velocity - balls[j].velocity;
			if (length(dist) < 2) {
				Vector3 vel2 = normalize(dist) * dot(normalize(dist), vel1);
				Vector3 u2 = balls[j].velocity + vel2;
				/*
				balls[i].velocity = balls[i].velocity + balls[j].velocity - u2;
				balls[j].velocity = u2;
				*/
				times[i] ++;
				times[j] ++;
				velocity_acc[i] += balls[i].velocity + balls[j].velocity - u2;
				velocity_acc[j] += u2;
			}
		}
	}

	float width = 5.0, height = 5.0;
	// Collision with walls
	for (int i = 0; i < SPHERES; i++) {
		if (balls[i].position.x < -width) {
			velocity_acc[i] += Vector3(fabs(balls[i].velocity.x), 0, balls[i].velocity.z);
			times[i] ++;
		}
		if (balls[i].position.x > width) {
			velocity_acc[i] += Vector3(-fabs(balls[i].velocity.x), 0, balls[i].velocity.z);
			times[i] ++;
		}
		if (balls[i].position.z < -height) {
			velocity_acc[i] += Vector3(balls[i].velocity.x, 0, fabs(balls[i].velocity.z));
			times[i] ++;
		}
		if (balls[i].position.z > height) {
			velocity_acc[i] += Vector3(balls[i].velocity.x, 0, -fabs(balls[i].velocity.z));
			times[i] ++;
		}
	}

	for (int i = 0; i < SPHERES; i++) {
		if (times[i] > 0) {
			balls[i].velocity = velocity_acc[i] / times[i];
		}
	}
	// Update position & orientation;
	for (int i = 0; i < SPHERES; i++) {
		Vector3 distance = balls[i].velocity * delta_time;
		balls[i].position += distance;
		Vector3 axis = normalize(Vector3(distance.z, 0, -distance.x));
		Quaternion rotation = Quaternion(axis, length(distance));
		if (length(distance) <= 0) {
			rotation = Quaternion::Identity();
		}
		balls[i].orientation = balls[i].orientation * rotation;
	}

	for (int i = 0; i < SPHERES; i++) {
		balls[i].position.to_array(cscene_host.position + 3 * i);
		balls[i].orientation.to_array(cscene_host.rotation + 4 * i);
	}
	
	do_gpu_raytracing();
	/*
	for (int i = 0; i < SPHERES; i++) {
		cscene_host.data[4 * SPHERES + 3 * i] = i * sin(time);
		cscene_host.data[4 * SPHERES + 3 * i + 2] = i * cos(time);
	}
	*/
	
}

void RaytracerApplication::render()
{
    int width, height;
    // query current window size, resize viewport
    get_dimension( &width, &height );
    glViewport( 0, 0, width, height );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // reset matrices
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

	if (!buffer) {
		buffer = new unsigned char [width * height * 4];
	}
	glColor4d( 1.0, 1.0, 1.0, 1.0 );
	glRasterPos2f( -1.0f, -1.0f );
	glDrawPixels( buf_width, buf_height, GL_RGBA,
		  GL_UNSIGNED_BYTE, &buffer[0] );
}

void RaytracerApplication::handle_event( const SDL_Event& event )
{
    int width, height;

    if ( !gpu_raytracing ) {
        camera_control.handle_event( this, event );
    }
	scene.handle_event(event);

    switch ( event.type )
    {
    case SDL_KEYDOWN:
        switch ( event.key.keysym.sym )
        {
		case KEY_RAYTRACE_GPU:
			do_gpu_raytracing();
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
	int width = DEFAULT_WIDTH;
	int height = DEFAULT_HEIGHT;
	buf_width = width;
	buf_height = height;
	if (!buffer) {
		buffer = new unsigned char [width * height * 4];
	}

	cscene.width = width;
	cscene.height = height;
	printf("CSC :%p\n", &cscene);
	gpu_raytracing = true;
	gpuErrchk(cudaMemcpy(cscene.data, cscene_host.data, sizeof(float) * 7 * SPHERES, cudaMemcpyHostToDevice));
	cudaRayTrace(&cscene, cimg);
	gpuErrchk(cudaMemcpy(buffer, cimg, 4 * dwidth * dheight, cudaMemcpyDeviceToHost));
}


}

using namespace _462;

static bool parse_args( Options* opt, int argc, char* argv[] )
{
    if ( argc < 2 ) {
		std::cout << "More arguments" << std::endl;
        return false;
    }

    opt->input_filename = argv[1];
    opt->output_filename = NULL;
    opt->open_window = true;
    opt->width = DEFAULT_WIDTH;
    opt->height = DEFAULT_HEIGHT;
    opt->num_samples = 1;
    for (int i = 2; i < argc; i++)
    {

    	if(strcmp(argv[i] + 1, "master") == 0) {    		
    		opt->master = true;
    		continue;
    	}

    	if(strcmp(argv[i] + 1, "slave") == 0) {
    		opt->slave = true;
    		opt->host = argv[i + 1]; // we assume the next parameter is the master's host for the slave to connect to
    		i++;
    		continue;
    	}

        switch (argv[i][1])
        {
        case 'd':
            if (i >= argc - 2) return false;
            opt->width = atoi(argv[++i]);
            opt->height = atoi(argv[++i]);
            // check for valid width/height
            if ( opt->width < 1 || opt->height < 1 )
            {
                std::cout << "Invalid window dimensions\n";
                return false;
            }
            break;
        case 'n':
            if (i < argc - 1)
                opt->num_samples = atoi(argv[++i]);
            break;
        }
    }

    return true;
}

using namespace std;
int main( int argc, char* argv[] )
{
    Options opt;
	int ret = 0;

    if ( !parse_args( &opt, argc, argv ) ) {
        return 1;
    }

    RaytracerApplication app( opt );

    // load the given scene
    if ( !load_scene( &app.scene, opt.input_filename ) ) {
        std::cout << "Error loading scene "
          << opt.input_filename << ". Aborting.\n";
        return 1;
    }

	Scene *scene = &app.scene;
	cout << "Geometries: " << scene->num_geometries() << endl;
	cout << "Meshes: " << scene->num_meshes() << endl;
	cout << "Materials: " << scene->num_materials() << endl;

	cout << "master:slave => " << opt.master << ":" << opt.slave << endl;

	if(opt.master) {
		Master::start();
	}else if(opt.slave) {
		Slave::start(opt.host);
	}	

	real_t fps = 20.0;
	const char* title = "15462 Project 3 - Raytracer";
	// start a new application
	ret = Application::start_application(&app,
					  opt.width,
					  opt.height,
					  fps, title);

	return ret;
}

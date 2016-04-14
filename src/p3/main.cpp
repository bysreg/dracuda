/**
 * @file main_rayracer.cpp
 * @brief Raytracer entry
 *
 * @author Eric Butler (edbutler)
 */


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

#include <SDL.h>

#include <stdlib.h>
#include <iostream>
#include <cstring>

cudaScene cscene;
cudaScene cscene_host;
using namespace std;

unsigned char *cimg;

namespace _462 {


//default size of the image in the window
#define DEFAULT_WIDTH 800
#define DEFAULT_HEIGHT 600
#define dwidth 800
#define dheight 600

#define BUFFER_SIZE(w,h) ( (size_t) ( 4 * (w) * (h) ) )

//key which starts the raytracer
#define KEY_RAYTRACE SDLK_r
//key which saves the current image to a file
#define KEY_SCREENSHOT SDLK_f
//key which runs the photon emitter
#define KEY_SEND_PHOTONS SDLK_t

#define KEY_START_ANIMATION SDLK_y

#define KEY_MOTION_BLUR SDLK_u

#define KEY_RAYTRACE_GPU SDLK_g


// pretty sure these are sequential, but use an array just in case
static const GLenum LightConstants[] = {
    GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3,
    GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7
};
static const size_t NUM_GL_LIGHTS = 8;

// renders a scene using opengl

/**
 * Struct of the program options.
 */
struct Options
{
    // whether to open a window or just render without one
    bool open_window;
    // not allocated, pointed it to something static
    const char* input_filename;
    // not allocated, pointed it to something static
    const char* output_filename;
	const char* animation_prefix;
    // window dimensions
    int width, height;
    int num_samples;
	bool motion_blur = false;
	bool animation = false;
};

class RaytracerApplication : public Application
{
public:
    void render_scene(const Scene& scene);

    RaytracerApplication( const Options& opt )
        : options( opt ), buffer( 0 ), buf_width( 0 ),
      buf_height( 0 ), queue_render_photon ( false ), raytracing( false ) { }

    virtual ~RaytracerApplication() {
		if (buffer)
			free( buffer );
		if (animation_buffer)
			free(animation_buffer);
	}

    virtual bool initialize();
    virtual void destroy();
    virtual void update( real_t );
    virtual void render();
    virtual void handle_event( const SDL_Event& event );

    // flips raytracing, does any necessary initialization
    void toggle_raytracing( int width, int height );
	void start_animation(int width, int height);
	void start_motion_blur(int width, int height);
	void output_animation( int width, int height , const char *prefix);
	void do_gpu_raytracing();

    // writes the current raytrace buffer to the output file
    void output_image();

    Raytracer raytracer;

    // the scene to render
    Scene scene;

    // options
    Options options;

    // the camera
    CameraRoamControl camera_control;

	bool pause;
	real_t speed;

    // the image buffer for raytracing
    unsigned char* buffer = 0;
	unsigned char* animation_buffer = 0;
	int frames;
    // width and height of the buffer
    int buf_width, buf_height;
    
    bool queue_render_photon;
// true if we are in raytrace mode.
    // if so, we raytrace and display the raytrace.
    // if false, we use normal gl rendering
    bool raytracing;
	bool gpu_raytracing;
    // false if there is more raytracing to do
    bool raytrace_finished;
	real_t animation_time = 0;
	bool  animation_finished = false;
};

bool RaytracerApplication::initialize()
{
    // copy camera into camera control so it can be moved via mouse
    camera_control.camera = scene.camera;
    bool load_gl = options.open_window;

	pause = false;
	speed = 1.0;

    try {

        Material* const* materials = scene.get_materials();
        Mesh* const* meshes = scene.get_meshes();

        // load all textures
        for ( size_t i = 0; i < scene.num_materials(); ++i ) {
            if ( !materials[i]->load() || ( load_gl && !materials[i]->create_gl_data() )) {
                std::cout << "Error loading texture, aborting.\n";
                return false;
            }
        }

        // load all meshes
        for ( size_t i = 0; i < scene.num_meshes(); ++i ) {
            if ( !meshes[i]->load() || ( load_gl && !meshes[i]->create_gl_data() ) ) {
                std::cout << "Error loading mesh, aborting.\n";
                return false;
            }
        }

    }
    catch ( std::bad_alloc const& )
    {
        std::cout << "Out of memory error while initializing scene\n.";
        return false;
    }
	// Alloc cuda mem
	int N = scene.num_geometries();
	int N_light = scene.num_lights();
	int N_material = scene.num_materials();

	gpuErrchk(cudaMalloc((void **)&cscene.position, sizeof(float) * 3 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.scale, sizeof(float) * 3 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.rotation, sizeof(float) * 4 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.type, sizeof(int) * 1 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.radius, sizeof(float) * 1 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.material, sizeof(float) * 1 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.light_pos, sizeof(float) * 3 * N_light));
	gpuErrchk(cudaMalloc((void **)&cscene.light_col, sizeof(float) * 3 * N_light));
	gpuErrchk(cudaMalloc((void **)&cscene.light_radius, sizeof(float) * 1 * N_light));
	gpuErrchk(cudaMalloc((void **)&cscene.ambient, sizeof(float) * 3 * N_material));
	gpuErrchk(cudaMalloc((void **)&cscene.diffuse, sizeof(float) * 3 * N_material));
	gpuErrchk(cudaMalloc((void **)&cscene.specular, sizeof(float) * 3 * N_material));
	gpuErrchk(cudaMalloc((void **)&cscene.vertex0, sizeof(float) * 3 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.vertex1, sizeof(float) * 3 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.vertex2, sizeof(float) * 3 * N));
	gpuErrchk(cudaMalloc((void **)&cscene.curand, sizeof(curandState) * dwidth * dheight));

	// Mirrored host mem
	cscene_host.position = (float *)malloc(sizeof(float) * 3 * N);
	cscene_host.rotation = (float *)malloc(sizeof(float) * 4 * N);
	cscene_host.scale = (float *)malloc(sizeof(float) * 3 * N);
	cscene_host.type = (int *)malloc(sizeof(int) * 1 * N);
	cscene_host.radius = (float *)malloc(sizeof(float) * 1 * N);
	cscene_host.material = (int *)malloc(sizeof(int) * 1 * N);
	cscene_host.light_pos = (float *)malloc(sizeof(float) * 3 * N_light);
	cscene_host.light_col = (float *)malloc(sizeof(float) * 3 * N_light);
	cscene_host.light_radius = (float *)malloc(sizeof(float) * 1 * N_light);
	cscene_host.ambient = (float *)malloc(sizeof(float) * 3 * N_material);
	cscene_host.diffuse = (float *)malloc(sizeof(float) * 3 * N_material);
	cscene_host.specular = (float *)malloc(sizeof(float) * 3 * N_material);
	cscene_host.vertex0 = (float *)malloc(sizeof(float) * 3 * N);
	cscene_host.vertex1 = (float *)malloc(sizeof(float) * 3 * N);
	cscene_host.vertex2 = (float *)malloc(sizeof(float) * 3 * N);
	

	for (size_t i = 0; i < N; i++) {
		Geometry *g = scene.get_geometries()[i];
		g->post_initialize();
		g->position.to_array(cscene_host.position + 3 * i);
		g->orientation.to_array(cscene_host.rotation + 4 * i);
		g->scale.to_array(cscene_host.scale + 3 * i);
		string type_string = typeid(*g).name();
			cout << type_string << endl;
		const Material *primary_material;
		if (type_string.find("Sphere") != string::npos) {
			Sphere *s = (Sphere *)g;
			primary_material = s->material;
			cscene_host.type[i] = 1;
		} else if (type_string.find("Triangle") != string::npos) {
			Triangle *t = (Triangle *)g;
			cscene_host.type[i] = 2;
			primary_material = t->vertices[0].material;
			t->vertices[0].position.to_array(cscene_host.vertex0 + 3 * i);
			t->vertices[1].position.to_array(cscene_host.vertex1 + 3 * i);
			t->vertices[2].position.to_array(cscene_host.vertex2 + 3 * i);
		} else if (type_string.find("Model") != string::npos) {
			Model *m = (Model *)m;
			primary_material = m->material;
			cscene_host.type[i] = 3;
		}
		for (int j = 0; j < N_material; j++) {
			if (scene.get_materials()[j] == primary_material)
				cscene_host.material[i] = j;
		}
	}

	scene.post_initialize();

	for (int i = 0; i < N_light; i++) {
		SphereLight l = scene.get_lights()[i];
		l.position.to_array(cscene_host.light_pos + 3 * i);
		l.color.to_array(cscene_host.light_col + 3 * i);
		cscene_host.light_radius[i] = l.radius;
	}

	for (int i = 0; i < N_material; i++)
	{
		Material *m = scene.get_materials()[i];
		m->ambient.to_array(cscene_host.ambient + 3 * i);
		m->diffuse.to_array(cscene_host.diffuse + 3 * i);
		m->specular.to_array(cscene_host.specular + 3 * i);
	}

	cudaMemcpy(cscene.position, cscene_host.position, sizeof(float) * 3 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.rotation, cscene_host.rotation, sizeof(float) * 4 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.scale, cscene_host.scale, sizeof(float) * 3 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.type, cscene_host.type, sizeof(int) * 1 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.radius, cscene_host.radius, sizeof(float) * 1 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.material, cscene_host.material, sizeof(int) * 1 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.light_pos, cscene_host.light_pos, sizeof(float) * 3 * N_light, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.light_col, cscene_host.light_col, sizeof(float) * 3 * N_light, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.light_radius, cscene_host.light_radius, sizeof(float) * 1 * N_light, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.ambient, cscene_host.ambient, sizeof(float) * 3 * N_material, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.diffuse, cscene_host.diffuse, sizeof(float) * 3 * N_material, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.specular, cscene_host.specular, sizeof(float) * 3 * N_material, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.vertex0, cscene_host.vertex0, sizeof(float) * 3 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.vertex1, cscene_host.vertex1, sizeof(float) * 3 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cscene.vertex2, cscene_host.vertex2, sizeof(float) * 3 * N, cudaMemcpyHostToDevice);

	
	scene.camera.position.to_array(cscene.cam_position);
	scene.camera.orientation.to_array(cscene.cam_orientation);
	scene.ambient_light.to_array(cscene.ambient_light_col);
	cscene.fov = scene.camera.fov;
	cscene.aspect = (dwidth + 0.0) / (dheight + 0.0);
	cscene.near_clip = scene.camera.near_clip;
	cscene.far_clip = scene.camera.far_clip;
	cscene.N = N;
	cscene.N_light = N_light;
	cscene.N_material = N_material;

    // set the gl state
    if ( load_gl ) {
        float arr[4];
        arr[3] = 1.0; // alpha is always 1

        glClearColor(
            scene.background_color.r,
            scene.background_color.g,
            scene.background_color.b,
            1.0f );

        scene.ambient_light.to_array( arr );
        glLightModelfv( GL_LIGHT_MODEL_AMBIENT, arr );

        const SphereLight* lights = scene.get_lights();

        for ( size_t i = 0; i < NUM_GL_LIGHTS && i < scene.num_lights(); i++ )
    {
            const SphereLight& light = lights[i];
            glEnable( LightConstants[i] );
            light.color.to_array( arr );
            glLightfv( LightConstants[i], GL_DIFFUSE, arr );
            glLightfv( LightConstants[i], GL_SPECULAR, arr );
            glLightf( LightConstants[i], GL_CONSTANT_ATTENUATION,
              light.attenuation.constant );
            glLightf( LightConstants[i], GL_LINEAR_ATTENUATION,
              light.attenuation.linear );
            glLightf( LightConstants[i], GL_QUADRATIC_ATTENUATION,
              light.attenuation.quadratic );
        }

        glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE );
    }
	// CUDA part
	gpuErrchk(cudaMalloc((void **)&cimg, 4 * dheight * dwidth));
	

    return true;
}

void RaytracerApplication::destroy()
{
}

void RaytracerApplication::update( real_t delta_time )
{
	if ( gpu_raytracing) {
		return ;
	}
    if ( raytracing ) {
        // do part of the raytrace
        if ( !raytrace_finished ) {
            assert( buffer );
            raytrace_finished = raytracer.raytrace( buffer, &delta_time );
        }
    } else {
        // copy camera over from camera control (if not raytracing)
        camera_control.update( delta_time );
        scene.camera = camera_control.camera;
    }

	// physics
	static const size_t NUM_ITER = 20;

	// step the physics simulation
	if (!pause) {
		real_t step_size = delta_time / NUM_ITER;
		for (size_t i = 0; i < NUM_ITER; i++) {
			scene.update(step_size * speed);
		}
	}

	animation_time += delta_time;
}

void RaytracerApplication::render()
{
    int width, height;

    // query current window size, resize viewport
    get_dimension( &width, &height );
    glViewport( 0, 0, width, height );

    // fix camera aspect
    Camera& camera = scene.camera;
    camera.aspect = real_t( width ) / real_t( height );

    // clear buffer
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // reset matrices
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

	if ( animation_finished ) {
        assert( animation_buffer );
		int cycles = animation_time / scene.animation_duration;
		int current_frame = (animation_time - cycles * scene.animation_duration) * scene.animation_fps;
        glColor4d( 1.0, 1.0, 1.0, 1.0 );
        glRasterPos2f( -1.0f, -1.0f );
		if (current_frame >= frames)
			current_frame = frames - 1;
		if (current_frame < 0)
			current_frame = 0;
        glDrawPixels( buf_width, buf_height, GL_RGBA,
              GL_UNSIGNED_BYTE, animation_buffer + current_frame * BUFFER_SIZE(buf_width, buf_height) );
	} else if ( raytracing || gpu_raytracing) {
        // if raytracing, just display the buffer
        assert( buffer );
        glColor4d( 1.0, 1.0, 1.0, 1.0 );
        glRasterPos2f( -1.0f, -1.0f );
        glDrawPixels( buf_width, buf_height, GL_RGBA,
              GL_UNSIGNED_BYTE, &buffer[0] );

    } else {
        // else, render the scene using opengl
        glPushAttrib( GL_ALL_ATTRIB_BITS );
        render_scene( scene );
        glPopAttrib();
    }
}

void RaytracerApplication::handle_event( const SDL_Event& event )
{
    int width, height;

    if ( !raytracing ) {
        camera_control.handle_event( this, event );
    }

	scene.handle_event(event);

    switch ( event.type )
    {
    case SDL_KEYDOWN:
        switch ( event.key.keysym.sym )
        {
        case KEY_RAYTRACE:
            get_dimension( &width, &height );
            toggle_raytracing( width, height );
            break;
		case KEY_RAYTRACE_GPU:
			do_gpu_raytracing();
			break;
        case KEY_SEND_PHOTONS:
            raytracer.initialize(&scene, options.num_samples, 0, 0);
            queue_render_photon=true;
			break;
        case KEY_SCREENSHOT:
            output_image();
            break;
		case KEY_START_ANIMATION:
            get_dimension( &width, &height );
			start_animation( width, height);
			animation_time = 0;
			animation_finished = true;
			break;
		case KEY_MOTION_BLUR:
			get_dimension(&width, &height);
			start_motion_blur(width, height);
			break;
		case SDLK_SPACE:
			pause = !pause;
			break;
        default:
            break;
        }
    default:
        break;
    }
}

void RaytracerApplication::start_motion_blur( int width, int height )
{
	if (!scene.animated) {
		std::cout << "The scene is not animated!" << std::endl;
		return;
	}
    assert( width > 0 && height > 0 );

	if (buffer)
		free(buffer);
	buffer = (unsigned char*) malloc (BUFFER_SIZE( width, height));
	unsigned int *accum_buffer = (unsigned int*) malloc(BUFFER_SIZE(width, height) * sizeof (unsigned int));
	frames = scene.animation_duration * scene.animation_fps;
	buf_width = width;
	buf_height = height;

	// initialize the raytracer (first make sure camera aspect is correct)
	scene.camera.aspect = real_t( width ) / real_t( height );

	if (!raytracer.initialize(&scene, options.num_samples, width, height)) {
		std::cout << "Raytracer initialization failed.\n";
		return; // leave untoggled since initialization failed.
	}

	for (unsigned int j = 0; j < BUFFER_SIZE(width, height); j++)
		accum_buffer[j] = 0;
	for (int i = 0; i < frames; i++) {
		double time = i / scene.animation_fps;
		scene.update_animation(time);
		raytracer.initialize(&scene, options.num_samples, width, height);
		std::cout << "Raytracing frame " << i << " / " << frames << std::endl;
		raytracer.raytrace(buffer, 0);
		for (unsigned int j = 0; j < BUFFER_SIZE(width, height); j++)
			accum_buffer[j] += buffer[j];
	}
	for (unsigned int j = 0; j < BUFFER_SIZE(width, height); j++)
		buffer[j] = accum_buffer[j] / frames;
	free(accum_buffer);
	raytracing = true;
	raytrace_finished = true;
}
void RaytracerApplication::start_animation( int width, int height )
{
	if (!scene.animated) {
		std::cout << "The scene is not animated!" << std::endl;
		return;
	}
    assert( width > 0 && height > 0 );

	frames = scene.animation_duration * scene.animation_fps;
	animation_buffer = (unsigned char*) malloc (BUFFER_SIZE( width, height) * frames);
	if ( !animation_buffer ) {
		std::cout << "Unable to allocate buffer.\n";
		return; // leave untoggled since we have no buffer.
	}
	buf_width = width;
	buf_height = height;

	// initialize the raytracer (first make sure camera aspect is correct)
	scene.camera.aspect = real_t( width ) / real_t( height );

	if (!raytracer.initialize(&scene, options.num_samples, width, height)) {
		std::cout << "Raytracer initialization failed.\n";
		return; // leave untoggled since initialization failed.
	}

	for (int i = 0; i < frames; i++) {
		double time = i / scene.animation_fps;
		scene.update_animation(time);
		raytracer.initialize(&scene, options.num_samples, width, height);
		std::cout << "Raytracing frame " << i << " / " << frames << std::endl;
		raytracer.raytrace(animation_buffer + i * BUFFER_SIZE(width, height), 0);
	}
	animation_finished = true;
}

void RaytracerApplication::output_animation( int width, int height , const char *prefix)
{
	if (!scene.animated) {
		std::cout << "The scene is not animated!" << std::endl;
		return;
	}
    assert( width > 0 && height > 0 );

	frames = scene.animation_duration * scene.animation_fps;
	if (buffer)
		free(buffer);
	buffer = (unsigned char*) malloc (BUFFER_SIZE( width, height));
	if ( !buffer ) {
		std::cout << "Unable to allocate buffer.\n";
		return; // leave untoggled since we have no buffer.
	}
	buf_width = width;
	buf_height = height;

	// initialize the raytracer (first make sure camera aspect is correct)
	scene.camera.aspect = real_t( width ) / real_t( height );

	if (!raytracer.initialize(&scene, options.num_samples, width, height)) {
		std::cout << "Raytracer initialization failed.\n";
		return; // leave untoggled since initialization failed.
	}

	char filename[1024];
	for (int i = 0; i < frames; i++) {
		double time = i / scene.animation_fps;
		scene.update_animation(time);
		raytracer.initialize(&scene, options.num_samples, width, height);
		std::cout << "Raytracing frame " << i << " / " << frames << std::endl;
		raytracer.raytrace(buffer, 0);
		sprintf(filename, "shots/%s%02d.png", prefix, i);
		imageio_save_image(filename, buffer, buf_width, buf_height);
	}
}
void RaytracerApplication::toggle_raytracing( int width, int height )
{
    assert( width > 0 && height > 0 );

    // do setup if starting a new raytrace
    if ( !raytracing ) {
		// only re-allocate if the dimensions changed
		if ( buf_width != width || buf_height != height ) {
			free( buffer );
			buffer = (unsigned char*) malloc( BUFFER_SIZE( width, height ) );
			if ( !buffer ) {
				std::cout << "Unable to allocate buffer.\n";
				return; // leave untoggled since we have no buffer.
			}
			buf_width = width;
			buf_height = height;
		}

		// initialize the raytracer (first make sure camera aspect is correct)
		scene.camera.aspect = real_t( width ) / real_t( height );

		if (!raytracer.initialize(&scene, options.num_samples, width, height)) {
			std::cout << "Raytracer initialization failed.\n";
			return; // leave untoggled since initialization failed.
		}

        // reset flag that says we are done
        raytrace_finished = false;
    }

    raytracing = !raytracing;
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
	gpu_raytracing = true;
	cudaRayTrace(&cscene, cimg);
        std::cout << "No image to output.\n";

	gpuErrchk(cudaMemcpy(buffer, cimg, 4 * dwidth * dheight, cudaMemcpyDeviceToHost));
}
void RaytracerApplication::output_image()
{
    static const size_t MAX_LEN = 256;
    const char* filename;
    char buf[MAX_LEN];

    if ( !buffer ) {
        std::cout << "No image to output.\n";
        return;
    }

    assert( buf_width > 0 && buf_height > 0 );

    filename = options.output_filename;

    // if we weren't given a file, use a default name
    if (!filename)
    {
        imageio_gen_name(buf, MAX_LEN);
        filename = buf;
    }

    if (imageio_save_image(filename, buffer, buf_width, buf_height))
    {
        std::cout << "Saved raytraced image to '" << filename << "'.\n";
    } else {
        std::cout << "Error saving raytraced image to '" << filename << "'.\n";
    }
}


void RaytracerApplication::render_scene(const Scene& scene)
{
    // backup state so it doesn't mess up raytrace image rendering
    glPushAttrib( GL_ALL_ATTRIB_BITS );
    glPushClientAttrib( GL_CLIENT_ALL_ATTRIB_BITS );

    glClearColor(
        scene.background_color.r,
        scene.background_color.g,
        scene.background_color.b,
        1.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glEnable( GL_NORMALIZE );
    glEnable( GL_DEPTH_TEST );
    glEnable( GL_TEXTURE_2D );

    // set camera transform

    const Camera& camera = scene.camera;

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( camera.get_fov_degrees(),
                    camera.get_aspect_ratio(),
                    camera.get_near_clip(),
                    camera.get_far_clip() );

    const Vector3& campos = camera.get_position();
    const Vector3 camref = camera.get_direction() + campos;
    const Vector3& camup = camera.get_up();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    gluLookAt( campos.x, campos.y, campos.z,
               camref.x, camref.y, camref.z,
               camup.x,  camup.y,  camup.z );

    // render each object
    if(queue_render_photon){
        queue_render_photon = false;
        raytracer.photonMap.update_photons();
    }
    raytracer.photonMap.render_photons();
    
    glEnable( GL_LIGHTING );
    // set light data
    float arr[4];
    arr[3] = 1.0; // w is always 1

    scene.ambient_light.to_array( arr );
    glLightModelfv( GL_LIGHT_MODEL_AMBIENT, arr );

    glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE );

    const SphereLight* lights = scene.get_lights();


    for ( size_t i = 0; i < NUM_GL_LIGHTS && i < scene.num_lights(); i++ )
    {
        const SphereLight& light = lights[i];
        glEnable( LightConstants[i] );
        light.color.to_array( arr );
        glLightfv( LightConstants[i], GL_DIFFUSE, arr );
        glLightfv( LightConstants[i], GL_SPECULAR, arr );
        glLightf( LightConstants[i], GL_CONSTANT_ATTENUATION,
          light.attenuation.constant );
        glLightf( LightConstants[i], GL_LINEAR_ATTENUATION,
          light.attenuation.linear );
        glLightf( LightConstants[i], GL_QUADRATIC_ATTENUATION,
          light.attenuation.quadratic );
        light.position.to_array( arr );
        glLightfv( LightConstants[i], GL_POSITION, arr );
    }

    
    Geometry* const* geometries = scene.get_geometries();

    for (size_t i = 0; i < scene.num_geometries(); ++i)
    {
        const Geometry& geom = *geometries[i];
        Vector3 axis;
        real_t angle;

        glPushMatrix();

        glTranslated(geom.position.x, geom.position.y, geom.position.z);
        geom.orientation.to_axis_angle(&axis, &angle);
        glRotated(angle*(180.0/PI), axis.x, axis.y, axis.z);
        glScaled(geom.scale.x, geom.scale.y, geom.scale.z);

        geom.render();

        glPopMatrix();
    }


    glPopClientAttrib();
    glPopAttrib();
}

} /* _462 */

using namespace _462;

/**
 * Prints program usage.
 */
static void print_usage( const char* progname )
{
    std::cout << "Usage: " << progname <<
    "input_scene [-n num_samples] [-r] [-d width"
    " height] [-o output_file]\n"
        "\n" \
        "Options:\n" \
        "\n" \
        "\t-r:\n" \
        "\t\tRaytraces the scene and saves to the output file without\n" \
        "\t\tloading a window or creating an opengl context.\n" \
        "\t-d width height\n" \
        "\t\tThe dimensions of image to raytrace (and window if using\n" \
        "\t\tand opengl context. Defaults to width=800, height=600.\n" \
        "\t-s input_scene:\n" \
        "\t\tThe scene file to load and raytrace.\n" \
        "\toutput_file:\n" \
        "\t\tThe output file in which to write the rendered images.\n" \
        "\t\tIf not specified, default timestamped filenames are used.\n" \
        "\n" \
        "Instructions:\n" \
        "\n" \
        "\tPress 'r' to raytrace the scene. Press 'r' again to go back to\n" \
        "\tgo back to OpenGL rendering. Press 'f' to dump the most recently\n" \
        "\traytraced image to the output file.\n" \
        "\n" \
        "\tUse the mouse and 'w', 'a', 's', 'd', 'q', and 'e' to move the\n" \
        "\tcamera around. The keys translate the camera, and left and right\n" \
        "\tmouse buttons rotate the camera.\n" \
        "\n" \
        "\tIf not using windowed mode (i.e., -r was specified), then output\n" \
        "\timage will be automatically generated and the program will exit.\n" \
        "\n";
}


/**
 * Parses args into an Options struct. Returns true on success, false on failure.
 */
static bool parse_args( Options* opt, int argc, char* argv[] )
{
    if ( argc < 2 ) {
        print_usage( argv[0] );
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
        case 'r':
            opt->open_window = false;
            break;
        case 'n':
            if (i < argc - 1)
                opt->num_samples = atoi(argv[++i]);
            break;
        case 'o':
            if (i < argc - 1)
                opt->output_filename = argv[++i];
			break;
		case 'p':
            if (i < argc - 1)
                opt->animation_prefix = argv[++i];
			break;
		case 'm':
			opt->motion_blur = true;
			break;
		case 'a':
			opt->animation = true;
			break;
        }
    }

    return true;
}

using namespace std;
int main( int argc, char* argv[] )
{
#ifdef OPENMP
    omp_set_num_threads(MAX_THREADS);
#endif

    
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
	
    // either launch a window or do a full raytrace without one,
    // depending on the option
    if ( opt.open_window ) {

        real_t fps = 60.0;
        const char* title = "15462 Project 3 - Raytracer";
        // start a new application
        ret = Application::start_application(&app,
                          opt.width,
                          opt.height,
                          fps, title);

    }
    else
    {
        app.initialize();

		if (opt.motion_blur) {
			app.start_motion_blur(opt.width, opt.height);
			app.output_image();
		} else if (opt.animation) {
			app.output_animation(opt.width, opt.height, opt.animation_prefix);
		} else {
			app.toggle_raytracing( opt.width, opt.height );
			if ( !app.raytracing ) {
				return 1; // some error occurred
			}
			assert( app.buffer );
			// raytrace until done
			app.raytracer.raytrace( app.buffer, 0 );
			app.output_image();
		}
		ret = 0;

    }
	return ret;
}

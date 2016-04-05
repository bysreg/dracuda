//
//  texture.cpp
//  
//
//  Created by Nathan A. Dobson on 2/1/15.
//
//

#include "scene/texture.hpp"

namespace _462{
const unsigned char* Texture::get_texture_data() const
{
    return data;
}

void Texture::get_texture_size( int* w, int* h ) const
{
    assert( w && h );
    *w = this->width;
    *h = this->height;
}

Color3 Texture::get_texture_pixel( int x, int y ) const
{
    if(width<=0 || height<=0){
        return Color3::White();
    }
    x=x%width;
    while(x<0){
        x+=width;
    }
    y=y%height;
    while(y<0){
        y+=height;
    }
    assert(x>=0 && y>=0 && x<width && y<height);
    return data ? Color3( data + 4 * (x + y * width) ) : Color3::White();
}

bool Texture::load(){
    // if no texture, nothing to do
    if ( filename.empty() )
        return true;
    
    std::cout << "Loading texture " << filename << "...\n";
    
    // allocates data with malloc
    data = imageio_load_image( filename.c_str(), &width, &height );
    
    if ( !data ) {
        std::cerr << "Cannot load texture file " << filename << std::endl;
        return false;
    }
    std::cout << "Finished loading texture" << std::endl;
    return true;
}

Color3 Texture::sample(Vector2 coord) const{
	return get_texture_pixel ((int)(coord.x * width), (int)(coord.y * height));
}

double Texture::sample_bump_u(Vector2 coord) const {
	int x = (int)(coord.x * width);
	int y = (int)(coord.y * height);
	Color3 c = get_texture_pixel (x + 1, y) - get_texture_pixel(x, y);
	return c.r;
}

double Texture::sample_bump_v(Vector2 coord) const {
	int x = (int)(coord.x * width);
	int y = (int)(coord.y * height);
	Color3 c = get_texture_pixel (x, y + 1) - get_texture_pixel(x, y);
	return c.r;
}

}

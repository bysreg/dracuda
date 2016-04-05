#include "scene/utils.hpp"

namespace _462 {
//quadratic formula

//If a solution exists, returns answers in x1 and x2, and returns true.
//Otherwise, returns false
bool solve_quadratic(real_t *x1,real_t *x2, real_t a, real_t b, real_t c){
    real_t b24ac = b*b-4*a*c;
    if(b24ac<0){
        return false;
    }else{
        real_t sb24ac=sqrt(b24ac);
        *x1=(-b+sb24ac)/(2*a);
        *x2=(-b-sb24ac)/(2*a);
        return true;
    }
}
//solve a quadratic equation, and then return the smallest solution larger than EPS
//if there is no solution, return -1
real_t solve_time(real_t a,real_t b,real_t c){
    real_t x1;
    real_t x2;
    if(solve_quadratic(&x1,&x2,a,b,c)){
        if(x1>EPS && x2>EPS){
            return std::min(x1,x2);
        }else if(x1>EPS){
            return x1;
        }else if(x2>EPS){
            return x2;
        }
    }
    return -1;
}

bool solve_time2(real_t a, real_t b, real_t c, real_t &x1, real_t &x2)
{
    if(solve_quadratic(&x1,&x2,a,b,c)){
		if (x1 > x2) {
			real_t temp = x1;
			x1 = x2;
			x2 = temp;
		}
		if (x2 < EPS) {
			return false;
		}
		if (x1 < EPS)
			x1 = EPS;
		return true;
    } else
		return false;
}
bool TriangleIntersection (Ray &ray, Vector3 v0, Vector3 v1, Vector3 v2, real_t &beta, real_t &gamma, real_t &time)
{
	real_t a = v0.x - v1.x;
	real_t b = v0.y - v1.y;
	real_t c = v0.z - v1.z;
	real_t d = v0.x - v2.x;
	real_t e = v0.y - v2.y;
	real_t f = v0.z - v2.z;
	real_t g = ray.d.x;
	real_t h = ray.d.y;
	real_t i = ray.d.z;
	real_t j = v0.x - ray.e.x;
	real_t k = v0.y - ray.e.y;
	real_t l = v0.z - ray.e.z;
	real_t ei_hf = e * i - h * f;
	real_t gf_di = g * f - d * i;
	real_t dh_eg = d * h - e * g;
	real_t M = a * ei_hf + b * gf_di + c * dh_eg;
	beta = (j * ei_hf + k * gf_di + l * dh_eg) / M;
	if (beta < 0 || beta > 1)
		return false;
	real_t ak_jb = a * k - j * b;
	real_t jc_al = j * c - a * l;
	real_t bl_kc = b * l - k * c;
	gamma = (i * ak_jb + h * jc_al + g * bl_kc) / M;
	if (gamma < 0 || gamma > 1)
		return false;
	if (1 - beta - gamma < 0)
		return false;
	time = -(f * ak_jb + e * jc_al + d * bl_kc) / M;
	if (time < EPS)
		return false;
	return true;
}
}

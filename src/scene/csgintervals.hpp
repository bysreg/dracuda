#ifndef _462_SCENE_CSG_INTERVALS_HPP_
#define _462_SCENE_CSG_INTERVALS_HPP_
#include <iostream>

#include <vector>
#include "math/math.hpp"
namespace _462 {

struct Interval
{
	real_t min;
	real_t max;
	int min_geometry;
	int max_geometry;
};


struct CSGIntervals
{
public:
	CSGIntervals();
	~CSGIntervals();
	std::vector<Interval> intervals;
	void add(Interval& interval);
	void clear();
	int size();
	Interval& operator [](const int index);
};

void intervals_union (CSGIntervals& interval0, CSGIntervals& interval1, CSGIntervals &result);
void intersection (CSGIntervals& interval0, CSGIntervals& interval1, CSGIntervals &result);
void difference (CSGIntervals& interval0, CSGIntervals& interval1, CSGIntervals &result);

}
#endif

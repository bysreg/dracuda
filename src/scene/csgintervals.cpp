#include "scene/csgintervals.hpp"
#include <iostream>

namespace _462 {

CSGIntervals::CSGIntervals()
{
}

CSGIntervals::~CSGIntervals()
{
}

std::ostream& operator<< (std::ostream &out, CSGIntervals& intervals)
{
	out << "{";
	int i = 0;
	for (i = 0; i < intervals.size() - 1; i++) {
		out << "[" << intervals[i].min << ", " << intervals[i].max << "], ";
	}
	if (intervals.size() > 0)
		out << "[" << intervals[i].min << ", " << intervals[i].max << "]";
	out << "}" << std::endl;
	return out;
}

void CSGIntervals::add(Interval& interval)
{
	intervals.push_back(interval);
}

void CSGIntervals::clear()
{
	intervals.clear();
}

Interval& CSGIntervals::operator [](const int index)
{
	return intervals[index];
}

int CSGIntervals::size()
{
	return intervals.size();
}

void intersection (CSGIntervals& interval0, CSGIntervals& interval1, CSGIntervals& result)
{
	int i = 0, j = 0, n0 = interval0.size(), n1 = interval1.size();
	Interval temp;
	while ((i < n0) && (j < n1)) {
		Interval &i0 = interval0[i];
		Interval &i1 = interval1[j];
		if (i0.max < i1.max)
		{
			// Intersect
			if (i0.max > i1.min) {
				temp.max = i1.max;
				temp.max_geometry = i1.max_geometry;
				if (i0.min < i1.min) {
					temp.min = i1.min;
					temp.min_geometry = i1.min_geometry;
				} else {
					temp.min = i0.min;
					temp.min_geometry = i0.min_geometry;
				}
				result.add(temp);
			}
			i++;
		} else {
			// Intersect
			if (i1.max > i0.min) {
				temp.max = i0.max;
				temp.max_geometry = i0.max_geometry;
				if (i0.min < i1.min) {
					temp.min = i1.min;
					temp.min_geometry = i1.min_geometry;
				} else {
					temp.min = i0.min;
					temp.min_geometry = i0.min_geometry;
				}
				result.add(temp);
			}
			j++;
		}
	}
}

void difference (CSGIntervals& interval0, CSGIntervals& interval1, CSGIntervals &result)
{
	int i = 0, j = 0, n0 = interval0.size(), n1 = interval1.size();
	Interval temp;
	Interval slice;
	if (n0 > 0)
		temp = interval0[0];
	else
		return;
	while ((i < n0) && (j < n1)) {
		Interval &i1 = interval1[j];
		if (i1.max < temp.max) {
			if (i1.max < temp.min) {
				j++;
				continue;
			} else {
				if (i1.min > temp.min) {
					// Slice
					slice.min = temp.min;
					slice.min_geometry = temp.min_geometry;
					slice.max = i1.min;
					slice.max_geometry = i1.min_geometry;
					result.add(slice);
				}
				// Truncate temp
				temp.min = i1.max;
				temp.min_geometry = i1.max_geometry;
				j++;
			}
		} else {
			if (i1.min > temp.min) {
				if (i1.min < temp.max) {
					// Slice
					slice.min = temp.min;
					slice.min_geometry = temp.min_geometry;
					slice.max = i1.min;
					slice.max_geometry = i1.min_geometry;
					result.add(slice);
				} else {
					result.add(temp);
				}
			}
			i++;
			if (i < n0) {
				temp = interval0[i];
			}
		}

	}
	if (i < n0) {
		result.add(temp);
		i++;
	}
	
	while (i < n0) {
		result.add(interval0[i]);
		i++;
	}
}

bool unite_interval(Interval &i0, Interval i1)
{
	if (i1.min > i0.max || i1.max < i0.min)
		return false;
	if (i1.max > i0.max) {
		i0.max = i1.max;
		i0.max_geometry = i1.max_geometry;
	}
	return true;
}

void intervals_union (CSGIntervals& interval0, CSGIntervals& interval1, CSGIntervals &result)
{
	int i = 0, j = 0, n0 = interval0.size(), n1 = interval1.size();
	Interval temp;
	bool temp_empty = true;
	while (1) {
		bool iln0 = i < n0;
		bool jln1 = j < n1;

		if (!iln0 && !jln1)
			break;
		if (temp_empty) {
			if (iln0 && jln1) {
				if (interval0[i].min < interval1[j].min) {
					temp = interval0[i];
					i++;
					temp_empty = false;
					continue;
				} else {
					temp = interval1[j];
					j++;
					temp_empty = false;
					continue;
				}
			} else if (iln0) {
				temp = interval0[i];
				i++;
				temp_empty = false;
				continue;
			} else if (jln1) {
				temp = interval1[j];
				j++;
				temp_empty = false;
				continue;
			}
		}
		if (iln0) {
			Interval &i0 = interval0[i];
			if (unite_interval(temp, i0)) {
				i++;
				continue;
			}
		}
		if (jln1) {
			Interval &i1 = interval1[j];
			if (unite_interval(temp, i1)) {
				j++;
				continue;
			}
		}
		result.add(temp);
		temp_empty = true;
	}
	if (!temp_empty)
		result.add(temp);
}

}


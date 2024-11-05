#ifndef VOLUME_H
#define VOLUME_H
#include <memory>
#include <cassert>

struct Volume{
	Volume()= default;
	Volume(int D, int W) : d(D), w(W), arr(new double[d*w*w]()){} 

	inline double& operator ()(int i, int j, int k) {
		assert(i<d && j<w && k<w);
		return arr[i*w*w + j*w + k];
	}
	inline double operator ()(int i, int j, int k) const {
		assert(i<d && j<w && k<w);
		return arr[i*w*w + j*w + k];
	}

	int d; // depth
	int w; // width and height
	
private:
	std::unique_ptr<double[]> arr;	
};

#endif

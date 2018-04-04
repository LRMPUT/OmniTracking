
#include <chrono>

#include "ParticleFilter.h"

using namespace std;

ParticleFilter::ParticleFilter(int iw, int ih, int icx, int icy, int inp)
 : opticalFlow(iw, ih, icx, icy), np(inp)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	gen.seed(seed);
}

void ParticleFilter::initFilter(cv::Mat prev, cv::Mat cur, cv::Rect iobj)
{
	uniform_int_distribution<int> uniDistTheta(0, 360 - 1);
	uniform_int_distribution<int> uniDistR(0, opticalFlow.getMaxR());

	for (int p = 0; p < np; ++p) {
		Particle part;
		part.theta = uniDistTheta(gen);
		part.r = uniDistR(gen);
		part.weight = 1.0 / np;
		particles.push_back(part);
	}
	
	cv::Mat rFlow, thetaFlow;
	opticalFlow.getFlowCart(prev, cur, rFlow, thetaFlow);

	obj = iobj;
	rFlowObj = rFlow(obj).clone();
	thetaFlowObj = thetaFlow(obj).clone();
}

Particle ParticleFilter::processImage(cv::Mat prev, cv::Mat cur)
{

	return Particle();
}

cv::Mat ParticleFilter::getPolar(cv::Mat img)
{
	return opticalFlow.cartToPolar(img);
}

float ParticleFilter::calcWeight(cv::Mat rFlowObj,
								cv::Mat thetaFlowObj,
								cv::Mat rFlowImg,
								cv::Mat thetaFlowImg,
								const Particle & part)
{
	return 0.0f;
}

void ParticleFilter::redraw()
{
	uniform_real_distribution<float> uniRealDist(0.0, 1.0);
	uniform_int_distribution<int> uniIntDist(0, particles.size() - 1);

	vector<Particle> newParticles;
	{
		int idx = uniIntDist(gen);
		float beta = 0;
		float maxW = 0;
		for (int e = 0; e < particles.size(); e++) {
			maxW = max(maxW, particles[e].weight);
		}
		for (int e = 0; e < particles.size(); e++) {
			double w = particles[idx].weight;
			beta += 2 * maxW*uniRealDist(gen);
			while (w < beta) {
				beta -= w;
				idx = (idx + 1) % particles.size();
				w = particles[idx].weight;
			}
			newParticles.push_back(particles[idx]);
		}
	}
	newParticles.swap(particles);
}

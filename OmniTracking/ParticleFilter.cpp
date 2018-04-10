
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
	cv::Mat rFlow, thetaFlow;
	opticalFlow.getFlowCart(prev, cur, rFlow, thetaFlow);

	disturbParticles();

	Particle bestParticle;
	float bestWeight = 0;
	for (auto it = particles.begin(); it != particles.end(); ++it) {
		float w = calcWeight(rFlowObj, thetaFlowObj, rFlow, thetaFlow, *it);
		it->weight = w;
		if (w > bestWeight) {
			bestWeight = w;
			bestParticle = *it;
		}
	}

	redraw();

	// mean
	double meanX = 0;
	double meanY = 0;
	for (auto it = particles.begin(); it != particles.end(); ++it) {
		int xVal = opticalFlow.xCoord(it->r, it->theta);
		int yVal = opticalFlow.yCoord(it->r, it->theta);
		meanX += xVal;
		meanY += yVal;
	}
	meanX /= particles.size();
	meanY /= particles.size();
	// variance
	double varX = 0;
	double varY = 0;
	for (auto it = particles.begin(); it != particles.end(); ++it) {
		int xVal = opticalFlow.xCoord(it->r, it->theta);
		int yVal = opticalFlow.yCoord(it->r, it->theta);
		varX += (xVal - meanX) * (xVal - meanX);
		varY += (yVal - meanY) * (yVal - meanY);
	}
	varX /= particles.size();
	varY /= particles.size();
	double meanR = opticalFlow.rCoord(meanX, meanY);
	double meanTheta = opticalFlow.thetaCoord(meanX, meanY);

	cout << "meanX = " << meanX << endl;
	cout << "meanY = " << meanY << endl;
	cout << "varX = " << varX << endl;
	cout << "varY = " << varY << endl;

	// if filter has converged then update reference object
	if (varX < 100 && varY < 100) {
		int objRSize = obj.width;
		int objThetaSize = obj.height;
		obj = cv::Rect(meanR - objRSize / 2,
			meanTheta - objThetaSize / 2,
			objRSize,
			objThetaSize);
		rFlowObj = rFlow(obj).clone();
		thetaFlowObj = thetaFlow(obj).clone();
	}

	{
		cv::Mat vis = opticalFlow.cartToPolar(cur);

		for (auto it = particles.begin(); it != particles.end(); ++it) {
			cv::circle(vis, cv::Point(meanR, meanTheta), 4, cv::Scalar(255, 0, 0));
		}

		cv::rectangle(vis, obj, cv::Scalar(0, 0, 255));

		cv::imshow("vis", vis);

		cv::waitKey();
	}
	return Particle{ (int)meanTheta, (int)meanR, 1.0 };
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
	double diffSumR = 0;
	double diffSumTheta = 0;
	int pixCnt = 0;
	for (int th = 0; th < thetaFlowObj.rows; ++th) {
		for (int r = 0; r < rFlowObj.cols; ++r) {
			int partTh = (part.theta - obj.height / 2 + th + 360) % 360;
			int partR = part.r - obj.width / 2 + r;
			if (partR >= 0 && partR <= opticalFlow.getMaxR()) {

				double diffR = rFlowObj.at<float>(th, r) - rFlowImg.at<float>(partTh, partR);
				double diffTheta = thetaFlowObj.at<float>(th, r) - thetaFlowImg.at<float>(partTh, partR);

				diffSumR += diffR * diffR;
				diffSumTheta += diffTheta * diffTheta;
				++pixCnt;
			}
		}
	}
	diffSumR /= pixCnt;
	diffSumTheta /= pixCnt;

	static constexpr double a = 0.1;
	return exp(-a * (diffSumR + diffSumTheta));
}

void ParticleFilter::disturbParticles()
{
	normal_distribution<float> normDistTheta(0, 3);
	normal_distribution<float> normDistR(0, 3);
	
	for (auto it = particles.begin(); it != particles.end(); ++it) {
		int newR = it->r + normDistR(gen);
		int newTheta = it->theta + normDistTheta(gen);

		newR = max(obj.width/2, newR);
		newR = min(opticalFlow.getMaxR() - (obj.width + 1) / 2, newR);
		newTheta %= 360;

		it->r = newR;
		it->theta = newTheta;
	}
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

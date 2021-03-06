#pragma once

#include <vector>
#include <random>

#include <opencv2/opencv.hpp>

#include "OpticalFlow.h"

struct Particle {
	int theta;
	int r;
	float weight;
};

class ParticleFilter
{
public:
	ParticleFilter(int iw, int ih, int icx, int icy, int inp);
	
	bool initFilter(cv::Mat prev, cv::Mat cur, cv::Rect iobj);

	Particle processImage(cv::Mat prev, cv::Mat cur);

	cv::Mat getPolar(cv::Mat img);
private:
	float calcWeight(cv::Mat rFlowObj,
					cv::Mat thetaFlowObj,
					cv::Mat rFlowImg,
					cv::Mat thetaFlowImg,
					const Particle &part);

	void getFlowObj(cv::Mat &rFlowObj,
					cv::Mat &thetaFlowObj,
					const cv::Mat &rFlowImg,
					const cv::Mat &thetaFlowImg,
					const cv::Rect &obj);

	void predict(const cv::Mat &rFlowImg,
				 const cv::Mat &thetaFlowImg);

	void disturbParticles();

	void redraw();


	OpticalFlow opticalFlow;

	cv::Rect obj;

	cv::Mat rFlowObj;

	cv::Mat thetaFlowObj;

	std::vector<Particle> particles;

	int np;

	std::default_random_engine gen;
};


#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

struct MovingPixel
{
	int x;
	int y;
	int r;
	int theta;
	double rFlow;
	double thetaFlow;
};

class OpticalFlow
{
public:
	OpticalFlow(int iw, int ih, int icx, int icy);

	std::vector<MovingPixel> getMovingPixels(cv::Mat prev, cv::Mat cur);

	void getFlowCart(cv::Mat prev,
					cv::Mat cur,
					cv::Mat rFlow,
					cv::Mat thetaFlow);

	cv::Mat cartToPolar(cv::Mat img);

	int getMaxR();
private:
	int rCoord(int x, int y);
	int thetaCoord(int x, int y);

	int xCoord(int r, int theta);
	int yCoord(int r, int theta);

	
	void getFlow(cv::Mat prevPolar,
		cv::Mat curPolar,
		cv::Mat rFlow,
		cv::Mat thetaFlow);

	int w, h, cx, cy;
	int maxR;

	cv::Mat polarToCartLookup;
};


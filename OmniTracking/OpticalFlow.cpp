#define _USE_MATH_DEFINES

#include <cmath>

#include "OpticalFlow.h"

using namespace std;

OpticalFlow::OpticalFlow(int iw, int ih, int icx, int icy)
	: w(iw), h(ih), cx(icx), cy(icy)
{
	maxR = std::min(h / 2.0, w / 2.0) - 1;

	polarToCartLookup = cv::Mat(360, maxR + 1, CV_32SC2, cv::Scalar(0, 0));
	for (int r = 0; r < maxR + 1; ++r) {
		for (int theta = 0; theta < 360; ++theta) {
			int x = xCoord(r, theta);
			int y = yCoord(r, theta);
			polarToCartLookup.at<cv::Vec2i>(theta, r)[0] = x;
			polarToCartLookup.at<cv::Vec2i>(theta, r)[1] = y;
		}
	}
}

std::vector<MovingPixel> OpticalFlow::getMovingPixels(cv::Mat prev, cv::Mat cur)
{
	cv::Mat prevPolar = cartToPolar(prev);
	cv::Mat curPolar = cartToPolar(cur);

	cv::Mat rFlow, thetaFlow;
	getFlow(prevPolar, curPolar, rFlow, thetaFlow);

	vector<MovingPixel> movingPixels;
	for (int theta = 0; theta < polarToCartLookup.rows; ++theta) {
		for (int r = 0; r < polarToCartLookup.cols; ++r) {
			if (abs(rFlow.at<float>(theta, r)) > 0.328 ||
				abs(thetaFlow.at<float>(theta, r)) > 0.328)
			{
				MovingPixel px;
				px.x = xCoord(r, theta);
				px.y = yCoord(r, theta);
				px.r = r;
				px.theta = theta;
				px.rFlow = rFlow.at<float>(theta, r);
				px.thetaFlow = thetaFlow.at<float>(theta, r);
				movingPixels.push_back(px);
			}
		}
	}

	return movingPixels;
}

void OpticalFlow::getFlowCart(const cv::Mat & prev, const cv::Mat & cur, cv::Mat & rFlow, cv::Mat & thetaFlow)
{
	cv::Mat prevPolar = cartToPolar(prev);
	cv::Mat curPolar = cartToPolar(cur);

	getFlow(prevPolar, curPolar, rFlow, thetaFlow);
}

int OpticalFlow::rCoord(int x, int y)
{
	return sqrt((x - cx)*(x - cx) + (y - cy)*(y - cy));
}

int OpticalFlow::thetaCoord(int x, int y)
{
	float theta = atan2(y - cy, x - cx) * 180 / M_PI;
	if (theta < 0) {
		theta += 360;
	}
	return (int)theta;
}

int OpticalFlow::xCoord(int r, int theta)
{
	int x = round(r * cos(theta * M_PI / 180.0)) + cx;
	return x;
}

int OpticalFlow::yCoord(int r, int theta)
{
	int y = round(r * sin(theta * M_PI / 180.0)) + cy;
	return y;
}

cv::Mat OpticalFlow::cartToPolar(cv::Mat img)
{
	cv::Mat polar(polarToCartLookup.size(), CV_8UC3, cv::Scalar(0, 0, 0));

	for (int theta = 0; theta < polarToCartLookup.rows; ++theta) {
		for (int r = 0; r < polarToCartLookup.cols; ++r) {
			int x = polarToCartLookup.at<cv::Vec2i>(theta, r)[0];
			int y = polarToCartLookup.at<cv::Vec2i>(theta, r)[1];
			polar.at<cv::Vec3b>(theta, r) = img.at<cv::Vec3b>(y, x);
		}
	}
	return polar;
}

int OpticalFlow::getMaxR()
{
	return maxR;
}

void OpticalFlow::getFlow(const cv::Mat & prevPolar, const cv::Mat & curPolar, cv::Mat & rFlow, cv::Mat & thetaFlow)
{
	cv::Mat prevPolarGray;
	cv::cvtColor(prevPolar, prevPolarGray, cv::COLOR_BGR2GRAY);
	cv::Mat curPolarGray;
	cv::cvtColor(curPolar, curPolarGray, cv::COLOR_BGR2GRAY);

	cv::Mat flow;
	cv::calcOpticalFlowFarneback(prevPolarGray, curPolarGray, flow, 0.4, 1, 12, 2, 8, 1.2, 0);
	rFlow = cv::Mat(flow.size(), CV_32FC1);
	thetaFlow = cv::Mat(flow.size(), CV_32FC1);
	for (int r = 0; r < flow.rows; ++r) {
		for (int c = 0; c < flow.cols; ++c) {
			rFlow.at<float>(r, c) = flow.at<cv::Vec2f>(r, c)[0];
			thetaFlow.at<float>(r, c) = flow.at<cv::Vec2f>(r, c)[1];
		}
	}
}

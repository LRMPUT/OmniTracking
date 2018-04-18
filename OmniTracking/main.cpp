#include <iostream>

#include <opencv2/opencv.hpp>

#include "ParticleFilter.h"

using namespace std;

cv::Point p1, p2, pCur;
bool active = false;
int picked = 0;

void mouseHandler(int event, int x, int y, int flags, void* param) {
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		/* left button clicked. ROI selection begins */
		p1 = cv::Point(x, y);
		active = true;
		picked = 1;
	}

	if (event == CV_EVENT_MOUSEMOVE)
	{
		/* mouse dragged. ROI being selected */
		if (!active && picked == 0) {
			p1 = cv::Point(x, y);
		}
		else if(active) {
			p2 = cv::Point(x, y);
		}
		pCur = cv::Point(x, y);
	}

	if (event == CV_EVENT_LBUTTONUP)
	{
		p2 = cv::Point(x, y);
		active = false;
		picked = 2;
	}
}

int main() {
	

	cv::VideoCapture cap("zyrafa.avi"); // open the default camera
	if (!cap.isOpened()) {
		cout << "video not opened" << endl;
		return -1;
	}

	cv::Mat frame1, frame2;

	int framesToSkip = 51;
	for (int i = 0; i < framesToSkip; ++i) {
		cap >> frame1;
	}

	cap >> frame1;
	cap >> frame2;
	
	int fw = frame1.cols;
	int fh = frame1.rows;
	cout << "fw = " << fw << endl;
	cout << "fh = " << fh << endl;
	
	ParticleFilter pf(fw, fh, fw / 2, fh / 2, 1000);

	cv::Mat frame2Polar = pf.getPolar(frame2);

	cv::namedWindow("pick", CV_WINDOW_AUTOSIZE);
	cv::imshow("pick", frame2Polar);
	cv::setMouseCallback("pick", mouseHandler, 0);

	for (;;) {
		cv::Mat curFrame = frame2Polar.clone();
		if (active || picked == 2) {
			cv::rectangle(curFrame, p1, p2, cv::Scalar(0, 0, 255));
		}
		if(!active) {
			cv::drawMarker(curFrame, pCur, cv::Scalar(0, 0, 255));
		}
		cv::imshow("pick", curFrame);

		if (cv::waitKey(10) >= 0) {
			break;
		}
	}
	cv::destroyWindow("pick");
	cout << "p1 = " << p1 << endl;
	cout << "p2 = " << p2 << endl;

	while (!pf.initFilter(frame1, frame2, cv::Rect(p1, p2))) {
		cap >> frame1;
		cap >> frame2;
	}

	cv::Mat prev = frame2;

	cv::namedWindow("image cart", 1);
	cv::namedWindow("image polar", 2);
	for (;;)
	{
		cv::Mat frame;
		cap >> frame; // get a new frame from camera
		if (!frame.empty()) {
			cv::Mat polar = pf.getPolar(frame);

			cv::imshow("image cart", frame);
			cv::imshow("image polar", polar);

			pf.processImage(prev, frame);
			if (cv::waitKey(30) >= 0) {
				break;
			}
			prev = frame;
		}
		else {
			break;
		}
	}
}

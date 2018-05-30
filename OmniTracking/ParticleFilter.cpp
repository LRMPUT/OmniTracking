
#include <chrono>
#include "ParticleFilter.h"
#include "Windows.h"


using namespace std;
using namespace cv;

int suma_roznic = 0;

Mat rFlowColor;
Mat thetaFlowColor;
float x_zielonckiego;
float y_zielonckiego;
int licznik = 0;

ParticleFilter::ParticleFilter(int iw, int ih, int icx, int icy, int inp)
 : opticalFlow(iw, ih, icx, icy), np(inp)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	gen.seed(seed);
}

bool ParticleFilter::initFilter(cv::Mat prev, cv::Mat cur, cv::Rect iobj)
{
	uniform_int_distribution<int> uniDistTheta(0, 360 - 1);
	uniform_int_distribution<int> uniDistR(obj.width / 2, opticalFlow.getMaxR() - (obj.width + 1) / 2);

	cv::Mat prevBlur;
	cv::Mat curBlur;
	cv::GaussianBlur(prev, prevBlur, cv::Size(0, 0), 0.8);
	cv::GaussianBlur(cur, curBlur, cv::Size(0, 0), 0.8);

	particles.clear();
	for (int p = 0; p < np; ++p) {
		Particle part;
		part.theta = uniDistTheta(gen);
		part.r = uniDistR(gen);
		part.weight = 1.0 / np;
		particles.push_back(part);
	}

	cv::Mat rFlow, thetaFlow;
	opticalFlow.getFlowCart(prevBlur, curBlur, rFlow, thetaFlow);

	//double minValR, maxValR;
	//cv::minMaxIdx(abs(rFlow), &minValR, &maxValR);
	//cout << "obj minValR = " << minValR << ", maxValR = " << maxValR << endl;
	//double minValTheta, maxValTheta;
	//cv::minMaxIdx(abs(thetaFlow), &minValTheta, &maxValTheta);
	//cout << "obj minValTheta = " << minValTheta << ", maxValTheta = " << maxValTheta << endl;

	cv::Scalar meanR = cv::mean(abs(rFlow));
	cv::Scalar meanTheta = cv::mean(abs(thetaFlow));
	cout << "meanR = " << meanR << " meanTheta = " << meanTheta << endl;

	if (meanR(0) > 0.05 && meanTheta(0) > 0.05) {
		obj = iobj;
		rFlowObj = rFlow(obj).clone();
		thetaFlowObj = thetaFlow(obj).clone();

		return true;
	}
	else {
		return false;
	}

}

Mat kolorowanie(Mat macierz)
{
	Mat macierz_kolorowa;
	macierz_kolorowa.create(macierz.rows, macierz.cols, CV_8UC3);

	for (int i = 0; i < macierz.rows; i++)
	{
		for (int j = 0; j < macierz.cols; j++)
		{
			if (macierz.at<float>(i, j) <= 0)
			{
				macierz_kolorowa.at<Vec3b>(i, j).val[0] = -(float)macierz.at<float>(i, j) / 1.5 * 255;
				macierz_kolorowa.at<Vec3b>(i, j).val[1] = 0;
				macierz_kolorowa.at<Vec3b>(i, j).val[2] = 0;
			}
			else
			{
				macierz_kolorowa.at<Vec3b>(i, j).val[0] = 0;
				macierz_kolorowa.at<Vec3b>(i, j).val[1] = 0;
				macierz_kolorowa.at<Vec3b>(i, j).val[2] = (float)macierz.at<float>(i, j) / 1.5 * 255;
			}
		}
	}

	return macierz_kolorowa;
}

Particle ParticleFilter::processImage(cv::Mat prev, cv::Mat cur)
{
	cv::Mat prevBlur;
	cv::Mat curBlur;
	cv::GaussianBlur(prev, prevBlur, cv::Size(0, 0), 0.8);
	cv::GaussianBlur(cur, curBlur, cv::Size(0, 0), 0.8);

	cv::Mat rFlow, thetaFlow;
	opticalFlow.getFlowCart(prevBlur, curBlur, rFlow, thetaFlow);

	//double minValR, maxValR;
	//cv::minMaxIdx(abs(rFlow), &minValR, &maxValR);
	//cout << "minValR = " << minValR << ", maxValR = " << maxValR << endl;
	//double minValTheta, maxValTheta;
	//cv::minMaxIdx(abs(thetaFlow), &minValTheta, &maxValTheta);
	//cout << "minValTheta = " << minValTheta << ", maxValTheta = " << maxValTheta << endl;

	cv::Scalar meanR = cv::mean(abs(rFlow));
	cv::Scalar meanTheta = cv::mean(abs(thetaFlow));
	cout << "meanR = " << meanR << " meanTheta = " << meanTheta << endl;

	if (meanR(0) > 0.05 && meanTheta(0) > 0.05) {
		//predict(rFlow, thetaFlow);

		disturbParticles();


		Particle bestParticle;
		float bestWeight = 0;
		int a = GetTickCount();
		for (auto it = particles.begin(); it != particles.end(); ++it) {
			float w = calcWeight(rFlowObj, thetaFlowObj, rFlow, thetaFlow, *it);
			it->weight = w;
			if (w > bestWeight) {
				bestWeight = w;
				bestParticle = *it;
			}
		}
		int b = GetTickCount();
		int roznica = b - a;
		cout << "roznica: " << roznica << endl;
		suma_roznic = suma_roznic + roznica;
		cout << "suma roznic:" << suma_roznic << endl;

		//for (auto it = particles.begin(); it != particles.end(); ++it) {

		//	cv::Mat vis = opticalFlow.cartToPolar(cur);

		//	cv::circle(vis, cv::Point(it->r, it->theta), 4, cv::Scalar(255, 0, 0));

		//	cout << "w = " << it->weight << endl;

		//	cv::imshow("rFlowObj", rFlowObj);
		//	cv::imshow("thetaFlowObj", thetaFlowObj);
		//	cv::imshow("rFlow", rFlow);
		//	cv::imshow("thetaFlow", thetaFlow);

		//	cv::imshow("vis", vis);

		//	cv::waitKey();
		//}

		rFlowColor.create(rFlow.rows, rFlow.cols, CV_8UC3);
		
		/*for (int i = 0; i < rFlow.rows; i++)
		{
			for (int j = 0; j < rFlow.cols; j++)
			{
				if (rFlow.at<float>(i, j)<=0)
				{
					rFlowColor.at<Vec3b>(i, j).val[0] = 0;
					rFlowColor.at<Vec3b>(i, j).val[1] = -rFlow.at<float>(i, j) / 2 * 255;
					rFlowColor.at<Vec3b>(i, j).val[2] = 0;
				}
				else
				{
					rFlowColor.at<Vec3b>(i, j).val[0] = 0;
					rFlowColor.at<Vec3b>(i, j).val[1] = 0;
					rFlowColor.at<Vec3b>(i, j).val[2] = rFlow.at<float>(i, j)/2*255;
				}
			}
		}

		thetaFlowColor.create(thetaFlow.rows, thetaFlow.cols, CV_8UC3);

		for (int i = 0; i < thetaFlow.rows; i++)
		{
			for (int j = 0; j < thetaFlow.cols; j++)
			{
				if (thetaFlow.at<float>(i, j) <= 0)
				{
					thetaFlowColor.at<Vec3b>(i, j).val[0] = 0;
					thetaFlowColor.at<Vec3b>(i, j).val[1] = -(float)thetaFlow.at<float>(i, j) / 1.5 * 255;
					thetaFlowColor.at<Vec3b>(i, j).val[2] = 0;
				}
				else
				{
					thetaFlowColor.at<Vec3b>(i, j).val[0] = 0;
					thetaFlowColor.at<Vec3b>(i, j).val[1] = 0;
					thetaFlowColor.at<Vec3b>(i, j).val[2] = (float)thetaFlow.at<float>(i, j) / 1.5 * 255;
				}
			}
		}*/

		//cv::imshow("rFlow", rFlow);
		/*cv::imshow("thetaFlowColor", thetaFlowColor);*/
		Mat thetaKolorowa = kolorowanie(thetaFlow);
		Mat rKolorowa = kolorowanie(rFlow);
		Mat rWzorzec = kolorowanie(rFlowObj);
		Mat thetaWzorzec = kolorowanie(thetaFlowObj);
		cv::imshow("thetaWzorzec", thetaWzorzec);
		imshow("thetaFlowColor", thetaKolorowa);
		cv::imshow("rWzorzec", rWzorzec);
		imshow("rFlowColor", rKolorowa);
		//cv::imshow("rFlowColor", rFlowColor);
		

		redraw();

		// mean
		double meanX = 0;
		double meanY = 0;
		vector<double> xVals;
		vector<double> yVals;
		vector<double> rVals;
		vector<double> thetaVals;
		for (auto it = particles.begin(); it != particles.end(); ++it) {
			//cout << "r = " << it->r << ", theta = " << it->theta << endl;

			int xVal = opticalFlow.xCoord(it->r, it->theta);
			int yVal = opticalFlow.yCoord(it->r, it->theta);
			meanX += xVal;
			meanY += yVal;
			xVals.push_back(xVal);
			yVals.push_back(yVal);
			rVals.push_back(it->r);
			thetaVals.push_back(it->theta);
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

		/*sort(xVals.begin(), xVals.end());
		sort(yVals.begin(), yVals.end());
		double medianX = xVals[xVals.size() / 2];
		double medianY = yVals[yVals.size() / 2];
		double medianR = opticalFlow.rCoord(medianX, medianY);
		double medianTheta = opticalFlow.thetaCoord(medianX, medianY);*/
		sort(rVals.begin(), rVals.end());
		sort(thetaVals.begin(), thetaVals.end());
		double medianR = rVals[rVals.size() / 2];
		double medianTheta = thetaVals[thetaVals.size() / 2];

		cout << "medianR = " << medianR << endl;
		cout << "medianTheta = " << medianTheta << endl;
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
				cv::circle(vis, cv::Point(it->r, it->theta), 4, cv::Scalar(255, 0, 0));
			}

			cv::rectangle(vis, obj, cv::Scalar(0, 0, 255));

			cv::Rect medianObj(medianR - obj.width/2,
							medianTheta - obj.height/2,
							obj.width,
							obj.height);

			//licznik++;
			//if(licznik == 15)
			//{ 
			//	x_zielonckiego = medianObj.x;
			//	y_zielonckiego = medianObj.y;

			//	Mat rFlowObj1(rFlow, Rect(x_zielonckiego, y_zielonckiego, obj.width, obj.height));
			//	rFlowObj = rFlowObj1.clone();

			//	Mat rThetaObj1(thetaFlow, Rect(x_zielonckiego, y_zielonckiego, obj.width, obj.height));
			//	thetaFlowObj = rThetaObj1.clone();
			//	licznik = 0;
			//}


			cv::rectangle(vis, medianObj, cv::Scalar(0, 255, 0));

			cv::imshow("vis", vis);

			cv::waitKey(10);
		}

		return Particle{ (int)meanTheta, (int)meanR, 1.0 };
	}
	else {
		return Particle{ 0, 0, -1.0 };
	}
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
	double maxRImg = 0;
	double maxThetaImg = 0;
	double maxRObj = 0;
	double maxThetaObj = 0;
	for (int th = 0; th < thetaFlowObj.rows; ++th) {
		for (int r = 0; r < thetaFlowObj.cols; ++r) {
			int partTh = (part.theta - obj.height / 2 + th + 360) % 360;
			int partR = part.r - obj.width / 2 + r;
			if (partR >= 0 && partR <= opticalFlow.getMaxR()) {

				maxRImg = max(maxRImg, (double)abs(rFlowImg.at<float>(partTh, partR)));
				maxRObj = max(maxRObj, (double)abs(rFlowObj.at<float>(th, r)));
				maxThetaImg = max(maxThetaImg, (double)abs(thetaFlowImg.at<float>(partTh, partR)));
				maxThetaObj = max(maxThetaObj, (double)abs(thetaFlowObj.at<float>(th, r)));
			}
		}
	}

	/*cout << "maxRImg = " << maxRImg << endl;
	cout << "maxThetaImg = " << maxThetaImg << endl;
	cout << "maxRObj = " << maxRObj << endl;
	cout << "maxThetaObj = " << maxThetaObj << endl;*/

	maxRImg = max(1.0, maxRImg);
	maxThetaImg = max(1.0, maxThetaImg);
	maxRObj = max(1.0, maxRObj);
	maxThetaObj = max(1.0, maxThetaObj);

	double diffSumR = 0;
	double diffSumTheta = 0;
	int pixCnt = 0;
	for (int th = 0; th < thetaFlowObj.rows; th=th+2) {
		for (int r = 0; r < thetaFlowObj.cols; r=r+2) {
			int partTh = (part.theta - obj.height / 2 + th + 360) % 360;
			int partR = part.r - obj.width / 2 + r;
			if (partR >= 0 && partR <= opticalFlow.getMaxR()) {

				double diffR = rFlowObj.at<float>(th, r)/maxRObj - rFlowImg.at<float>(partTh, partR)/maxRImg;
				double diffTheta = thetaFlowObj.at<float>(th, r)/maxThetaObj - thetaFlowImg.at<float>(partTh, partR)/maxThetaImg;

				diffSumR += diffR * diffR;
				diffSumTheta += diffTheta * diffTheta;
				++pixCnt;
			}
		}
	}
	diffSumR /= pixCnt;
	diffSumTheta /= pixCnt;

	static constexpr double a = 10.0;
	return exp(-a * (diffSumR + diffSumTheta));
}

void ParticleFilter::getFlowObj(cv::Mat & rFlowObj,
								cv::Mat & thetaFlowObj,
								const cv::Mat & rFlowImg,
								const cv::Mat & thetaFlowImg,
								const cv::Rect & curObj)
{
	for (int th = 0; th < thetaFlowObj.rows; ++th) {
		for (int r = 0; r < thetaFlowObj.cols; ++r) {
			int partTh = (curObj.y - th + 360) % 360;
			int partR = curObj.x + r;
			if (partR >= 0 && partR <= opticalFlow.getMaxR()) {
				rFlowObj.at<float>(th, r) = rFlowImg.at<float>(partTh, partR);
				thetaFlowObj.at<float>(th, r) = thetaFlowImg.at<float>(partTh, partR);
			}
		}
	}
}

void ParticleFilter::predict(const cv::Mat &rFlowImg,
							const cv::Mat &thetaFlowImg)
{
	for (auto it = particles.begin(); it != particles.end(); ++it) {
		double meanR = 0;
		double meanTheta = 0;
		int pixCnt = 0;
		for (int th = 0; th < obj.height; ++th) {
			for (int r = 0; r < obj.width; ++r) {
				int partTh = (it->theta - obj.height / 2 + th + 360) % 360;
				int partR = it->r - obj.width / 2 + r;
				if (partR >= 0 && partR <= opticalFlow.getMaxR()) {
					meanR += rFlowImg.at<float>(partTh, partR);
					meanTheta += thetaFlowImg.at<float>(partTh, partR);
					++pixCnt;
				}
			}
		}
		meanR /= pixCnt;
		meanTheta /= pixCnt;
		/*if (abs(meanR) > 0.1 || abs(meanTheta) > 0.1) {
			cout << "meanR = " << meanR << endl;
			cout << "meanTheta = " << meanTheta << endl;
		}*/

		it->r += 5 * meanR;
		it->theta += 5 * meanTheta;
	}
}

void ParticleFilter::disturbParticles()
{
	normal_distribution<float> normDistTheta(0, 6);
	normal_distribution<float> normDistR(0, 6);
	
	for (auto it = particles.begin(); it != particles.end(); ++it) {
		int newR = it->r + normDistR(gen);
		int newTheta = it->theta + normDistTheta(gen);

		newR = max(obj.width/2, newR);
		newR = min(opticalFlow.getMaxR() - (obj.width + 1) / 2, newR);
		/*newR = max(0, newR);
		newR = min(opticalFlow.getMaxR(), newR);*/
		newTheta = (newTheta + 360) % 360;

		it->r = newR;
		it->theta = newTheta;
	}
}

void ParticleFilter::redraw()
{
	uniform_real_distribution<float> uniRealDist(0.0, 1.0);
	uniform_int_distribution<int> uniIntDist(0, particles.size() - 1);

	uniform_int_distribution<int> uniDistTheta(0, 360 - 1);
	uniform_int_distribution<int> uniDistR(obj.width / 2, opticalFlow.getMaxR() - (obj.width + 1) / 2);

	vector<Particle> newParticles;
	{
		int idx = uniIntDist(gen);
		float beta = 0;
		float maxW = 0;
		for (int e = 0; e < particles.size(); e++) {
			maxW = max(maxW, particles[e].weight);
		}
		int nDrawPart = 0.95 * particles.size();
		for (int e = 0; e < nDrawPart; e++) {
			double w = particles[idx].weight;
			beta += 2 * maxW*uniRealDist(gen);
			while (w < beta) {
				beta -= w;
				idx = (idx + 1) % particles.size();
				w = particles[idx].weight;
			}
			newParticles.push_back(particles[idx]);
		}

		for (int p = 0; p < particles.size() - nDrawPart; ++p) {
			Particle part;
			part.theta = uniDistTheta(gen);
			part.r = uniDistR(gen);
			part.weight = 1.0 / particles.size();
			newParticles.push_back(part);
		}
	}
	newParticles.swap(particles);
}

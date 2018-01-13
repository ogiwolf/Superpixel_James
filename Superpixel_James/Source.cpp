#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include <direct.h>
#include <time.h>
#include <ctime>

#define PI  3.14159265
#define SP 50

using namespace cv;
using namespace std;

Mat src;

vector<int> label;

struct  meta
{
	Vec3b color;
	double sim;
	int spnum;
};
struct superpixel{
	Rect rectangle;
	Mat roiyvalue;
	Mat roixvalue;
	Mat mask;

	int label;
	int size;
	double var;
	int layer;
	bool flag;
};

struct avg_color_
{
	int r;
	int g;
	int b;
	int cnt;
};

struct LAB{    // double chage to  float
	float L;
	float A;
	float B;
};

void firstseam(vector <vector<int>> &superpixels, Mat &energyx, Mat &energyy, vector <vector<int>> &rst, vector <meta> &metaout, vector <superpixel> &supout);
double gaussian(int x, int y);


uchar *src_labptr;

int main(int atgc, char **argv){

	src = imread("Lenna.png", 1);
	imshow("src", src);
	cout << "Input image" << endl;

	vector	<vector<int>> superpixels;
	vector <vector<int>>  rst;

	//***** Convert to LAB ******
	clock_t start_tag = clock();

	Mat src_lab;
	src_lab = src.clone(); //OpenCV�v���ƻs
	cvtColor(src_lab, src_lab, CV_BGR2Lab);

	clock_t end_tag = clock();

	cout << "Time eplased LAB :" << (double)(end_tag - start_tag) / CLOCKS_PER_SEC << endl;
	//*****************************

	src_labptr = src_lab.data;

	//***** K means***************
	start_tag = clock();

	Mat samples(src.cols*src.rows, 3, CV_32F);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(i + j*src.rows, z) = src.at<Vec3b>(i, j)[z];

	int clusterCount = 8; //���O��
	int attempts = 10;    //����10��
	Mat klabels;	      //�]�X���G
	Mat centers;
	kmeans(samples, clusterCount, klabels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 1.0), attempts, KMEANS_RANDOM_CENTERS, centers);
	Mat Kimage(src.size(), src.type());
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			int cluster_idx = klabels.at< int>(i + j*src.rows, 0);
			Kimage.at< Vec3b>(i, j)[0] = centers.at< float>(cluster_idx, 0);
			Kimage.at< Vec3b>(i, j)[1] = centers.at< float>(cluster_idx, 1);
			Kimage.at< Vec3b>(i, j)[2] = centers.at< float>(cluster_idx, 2);
		}
	}
	imshow("kimage", Kimage);

	cvtColor(Kimage, Kimage, CV_BGR2GRAY);

	end_tag = clock();
	cout << "Time eplased K means :" << (double)(end_tag - start_tag) / CLOCKS_PER_SEC << endl;
	//************************

	Mat lab[3];
	split(src_lab, lab);

	Mat clustersobelx, clustersobely;
	Mat cannyedge;
	Mat sobelrstx, sobelrsty;
	Mat sobel_lab_x[3], sobel_lab_y[3];
	start_tag = clock();
	GaussianBlur(src_lab, src_lab, Size(3, 3), 0, 0);
	Canny(src_lab, cannyedge, 50, 250, 3, false);
	imshow("Canny", cannyedge);

	Sobel(src_lab, sobelrstx, CV_32F, 0, 1, 3, 1);
	Sobel(src_lab, sobelrsty, CV_32F, 1, 0, 3, 1);
	convertScaleAbs(sobelrstx, sobelrstx);
	convertScaleAbs(sobelrsty, sobelrsty);
	split(sobelrstx, sobel_lab_x);
	split(sobelrsty, sobel_lab_y);

	imshow("Soble_x", sobelrstx);
	imshow("Sobel_y", sobelrsty);

	end_tag = clock();
	cout << "Time elapsed edge :" << (double)(end_tag - start_tag) / CLOCKS_PER_SEC << endl;


	GaussianBlur(Kimage, Kimage, Size(3, 3), 0, 0);
	Sobel(Kimage, clustersobelx, CV_32F, 0, 1, 3, 1);
	Sobel(Kimage, clustersobely, CV_32F, 1, 0, 3, 1);
	convertScaleAbs(clustersobelx, clustersobelx);
	convertScaleAbs(clustersobely, clustersobely);

	Mat energy_x, energy_y;
	energy_x.create(src.size(), CV_32F);
	energy_y.create(src.size(), CV_32F);
	float *ptr_energy_x = energy_x.ptr<float>();
	float *ptr_energy_y = energy_y.ptr<float>();
	uchar *ptr_sobel_x = sobelrstx.ptr<uchar>();
	uchar *ptr_sobel_y = sobelrsty.ptr<uchar>();
	uchar *ptr_canny = cannyedge.ptr<uchar>();

	//sobel + canny
	for (int i = 0; i < src.cols*src.rows; i++) {
		ptr_energy_y[i] = ((float)abs(ptr_sobel_y[i * 3]) + (float)abs(ptr_sobel_y[i * 3 + 1]) + (float)abs(ptr_sobel_y[i * 3 + 2]) + 0.25 * (float)ptr_canny[i]);
		ptr_energy_x[i] = ((float)abs(ptr_sobel_x[i * 3]) + (float)abs(ptr_sobel_x[i * 3 + 1]) + (float)abs(ptr_sobel_x[i * 3 + 2]) + 0.25 * (float)ptr_canny[i]);
	}
	imshow("sobelx+canny", energy_x);
	imshow("sobely+canny", energy_y);

	/*
	Mat sobelrst,sobelrsttrans;
	transpose(sobelrst, sobelrsttrans);	//Transposes a matrix.
	flip(sobelrsttrans, sobelrsttrans, 3); //Flips a 2D array around vertical, horizontal, or both axes.
	*/



	//*******dump label and draw seam on src pic 

	vector<vector<int>> original;
	vector<int> initial;
	vector <meta> outmeta;
	vector <superpixel> supout;
	//vector<vector<int>> rst;

	for (int x = 0; x < 1; x++) {
		original.push_back(initial);
		for (int y = 0; y < src.cols*src.rows; y++) {
			original.at(x).push_back(y);
		}
	}

	firstseam(original, energy_x, energy_y, rst, outmeta, supout);


	waitKey(0);
}

void firstseam(vector <vector<int>> &superpixels, Mat &energyx, Mat &energyy, vector <vector<int>> &rst, vector <meta> &metaout, vector<superpixel> &supout){

	vector <Rect> sprectx, sprecty;	//(int x, int y, int width, int height)
	vector <avg_color_> avgcolor;
	avgcolor.reserve(5000);
	avgcolor.resize(5000);
	Mat tempcolor;
	tempcolor.create(src.size(), CV_32F);
	supout.reserve(10000);
	supout.resize(10000);

	Mat roi, weimap;
	//Mat mask;
	//uchar *ptrmask;
	float *ptr_roi;
	int col, row;

	vector <float> weightmap_x;
	vector <float> weightmap_y;
	vector <float> dynamic_map_x_value;	//roi value
	vector <float> dynamic_map_y_value;
	vector <int> dynamic_seam_x;		//seam position
	vector <int> dynamic_seam_y;
	vector<vector<int>> dynamic_seam_sp;//store rst

	dynamic_seam_sp.resize(src.cols / SP + src.rows / SP);

	//******************y****************
	int y_block;
	y_block = src.cols / SP;

	int first_block, final_block;
	first_block = final_block = (src.cols%SP) / 2;

	if ((src.cols%SP) % 2 != 0) first_block++;

	sprecty.push_back(Rect(0, 0, first_block + SP, src.rows));
	for (int i = 1; i < y_block - 1; i++){
		sprecty.push_back(Rect(i*SP + first_block, 0, SP, src.rows));
	}
	if (final_block != 0) sprecty.push_back(Rect(src.cols - (final_block + SP), 0, final_block + SP, src.rows));
	else sprecty.push_back(Rect(src.cols - SP, 0, SP, src.rows));

	for (int x = 0; x < sprecty.size(); x++){

		double center = gaussian(0, sprecty.at(x).width); // meaning?
		weightmap_y.resize(sprecty.at(x).width);
		for (int i = -sprecty.at(x).width / 2; i < sprecty.at(x).width / 2; i++)
			weightmap_y.at(sprecty.at(x).width / 2 + i) = gaussian(i, sprecty.at(x).width) / center;  // ����weight�v��++ ?

		energyy(sprecty[x]).copyTo(roi); 
		/*mask = Mat::ones(roi.size(), CV_8UC1);
		for (int y = 0; y < superpixels.at(x).size(); y++){
			col = (superpixels.at(x).at(y) % src.cols) - sprecty[x].x;
			row = (superpixels.at(x).at(y) / src.cols) - sprecty[x].y;
			mask.at<uchar>(row, col) = 255;
		}*/

		roi.convertTo(roi, CV_32FC1);
		//ptrmask = mask.data;
		ptr_roi = roi.ptr<float>();
		dynamic_map_y_value.resize(sprecty.at(x).height*sprecty.at(x).width);
		dynamic_seam_y.resize(sprecty.at(x).height);

		for (int i = 0; i < sprecty.at(x).height; i++){
			for (int j = 0; j < sprecty.at(x).width; j++){
				ptr_roi[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] * weightmap_y[j];
			}
		}

		for (int i = 0; i < sprecty.at(x).height; i++) {
			for (int j = 0; j < sprecty.at(x).width; j++) {
				//First row value from weight roi
				if (i == 0) {
					dynamic_map_y_value[j] = ptr_roi[j];
					//cout << "dynamic_map_y_value[j]" << dynamic_map_y_value[j] << endl;
				}
				//Following rows find the largest from above connected to itself
				else {

					if (j == 0) {	//left boundary
						if (dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j] > dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1])
							dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j];
						else
							dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1];
					}
					else if (j == sprecty.at(x).width) { //right  boundary
						if (dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1] > dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j])
							dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1];
						else
							dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j];
					}
					else {	// middle 
						if (dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1] > dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j]){
							if (dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1] > dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1])
								dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1];
							else
								dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1];
						}
						else{
							if (dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j] > dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1])
								dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j];
							else
								dynamic_map_y_value[i*sprecty.at(x).width + j] = ptr_roi[i*sprecty.at(x).width + j] + dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1];
						}
					}
				}
			}
		}


		
		for(int j = 0; j < sprecty.at(x).width; j++  ){
			
		}

	}



}




double gaussian(int x, int w) {

	const double EulerConstant = std::exp(1.0);
	double c = 1;
	double b = 0;
	double a = 1 / (c*sqrt(2 * 3.14159265358979323846));
	double s = (double)w / 5;
	double p = 0 - (x / s - b)*(x / s - b) / 2 * c*c;
	return a*pow(EulerConstant, p);

}
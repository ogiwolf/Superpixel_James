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
	Vec3b avgcolor;
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
void superpixelsplit(vector<vector<int>> sp, vector <Rect> &roirect, Mat & energy_y, Mat & energy_x, vector <vector<int>> &rst, vector <meta> &meta, vector <superpixel> &sup);
void iteratorseam(vector <vector<int>> &superpixels, Mat &energyx, Mat &energyy, vector <vector<int>> &rst, vector <meta> &meta, vector <superpixel> &sup);
double gaussian(int x, int y);
double simil(Vec3b a, Vec3b b);


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
	vector<vector<int>> rst1, rst2, rst3, rst4, rst5, rst6;

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

		double center = gaussian(0, sprecty.at(x).width); 
		weightmap_y.resize(sprecty.at(x).width);
		for (int i = -sprecty.at(x).width / 2; i < sprecty.at(x).width / 2; i++)
			weightmap_y.at(sprecty.at(x).width / 2 + i) = gaussian(i, sprecty.at(x).width) / center;  

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


		int j;
		for (j = 0; j < sprecty.at(x).width; j++){
			if (dynamic_map_y_value[(sprecty.at(x).height - 1)*sprecty.at(x).width + dynamic_seam_y.at(sprecty.at(x).height - 1)] < dynamic_map_y_value[(sprecty.at(x).height - 1)*sprecty.at(x).width + j]) {
				dynamic_seam_y.at(sprecty.at(x).height - 1) = j;	
			}
		}

		for (int i = sprecty.at(x).height - 2; i >= 0; i--){
			j = dynamic_seam_y.at(i + 1);
			if (j == 0){
				if (dynamic_map_y_value[i*sprecty.at(x).width + j] <= dynamic_map_y_value[i*sprecty.at(x).width + j + 1]) {					
					dynamic_seam_y.at(i) = j + 1;					
				}
				else {
					dynamic_seam_y.at(i) = j;					
				}
			}
			else if (j == sprecty.at(x).width - 1){
				if (dynamic_map_y_value[i*sprecty.at(x).width + j] <= dynamic_map_y_value[i*sprecty.at(x).width + j - 1]) {
					dynamic_seam_y.at(i) = j - 1;					
				}
				else {
					dynamic_seam_y.at(i) = j;
				}
			}
			else {
				if (dynamic_map_y_value[i*sprecty.at(x).width + j] < dynamic_map_y_value[i*sprecty.at(x).width + j + 1]) {
					if (dynamic_map_y_value[i*sprecty.at(x).width + j - 1] > dynamic_map_y_value[i*sprecty.at(x).width + j + 1]) {
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						dynamic_seam_y.at(i) = j - 1;
						//else continue;
					}
					else {
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						dynamic_seam_y.at(i) = j + 1;
						//else continue;
					}
				}
				else {
					if (dynamic_map_y_value[i*sprecty.at(x).width + j - 1] > dynamic_map_y_value[i*sprecty.at(x).width + j]) {
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						dynamic_seam_y.at(i) = j - 1;
						//else continue;
					}
					else {
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						dynamic_seam_y.at(i) = j;
						//else continue;
					}
				}
			}
		}

		//cout << "x :" << x << endl;
		dynamic_seam_sp.at(x) = (dynamic_seam_y);
		dynamic_seam_y.clear();
		dynamic_map_y_value.clear();
	}


	//********************************************************************************


	//*********x**********************************************************************
	int x_block;
	x_block = src.rows / SP;

	first_block = (src.rows%SP) / 2;
	final_block = (src.rows%SP) / 2;
	
	if ((src.rows%SP) % 2 != 0) 
		first_block++;

	sprectx.push_back(Rect(0, 0, src.cols, first_block + SP));
	for (int i = 1; i < x_block - 1; i++){
		sprectx.push_back(Rect(0, i*SP + first_block, src.cols, SP));
	}
	if (final_block != 0) sprectx.push_back(Rect(0, src.rows - (final_block + SP), src.cols,final_block+ SP));
	else sprectx.push_back(Rect(0, src.rows - SP, src.cols, SP));


	for (int x = 0; x < sprectx.size(); x++){
		double center = gaussian(0, sprectx.at(x).height);
		weightmap_x.resize(sprectx.at(x).height);

		for (int i = -sprectx.at(x).height / 2; i < sprectx.at(x).height / 2; i++){
			weightmap_x.at(sprectx.at(x).height / 2 + i) = gaussian(i, sprectx.at(x).height) / center;
		}
		energyx(sprectx[x]).copyTo(roi);
		roi.convertTo(roi, CV_32F);
		ptr_roi = roi.ptr<float>();
		
		dynamic_map_x_value.resize(sprectx.at(x).height*sprectx.at(x).width);
		dynamic_seam_x.resize(sprectx.at(x).width);

		for (int i = 0; i < sprectx.at(x).height; i++) {
			for (int j = 0; j < sprectx.at(x).width; j++) {
				ptr_roi[i*sprectx.at(x).width + j] = ptr_roi[i*sprectx.at(x).width + j];// *weightmap_x[i];
			}
		}

		for (int i = 0; i < sprectx.at(x).width; i++) {
			for (int j = 0; j < sprectx.at(x).height; j++) {

				//First col value from weight roi
				if (i == 0) {

					dynamic_map_x_value[j*sprectx.at(x).width] = ptr_roi[j*sprectx.at(x).width];
					//cout << "Dynamic_map_x_value[j*roirect.at(x).width]" << Dynamic_map_x_value[j*roirect.at(x).width] << endl;

				}
				//Following cols find the largest from above connected to itself
				else {
					if (j == 0) {	//top boundary
						if (dynamic_map_x_value[j*sprectx.at(x).width + i - 1] > dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1])
							dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[j*sprectx.at(x).width + i - 1];
						else
							dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1];
					}
					else if (j == sprectx.at(x).height - 1) { //down  boundary
						if (dynamic_map_x_value[j*sprectx.at(x).width + i - 1] > dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1])
							dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[j*sprectx.at(x).width + i - 1];
						else
							dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1];
					}
					else {	// middle 
						if (dynamic_map_x_value[j*sprectx.at(x).width + i - 1] < dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1])
							if (dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1] < dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1])
								dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1];
							else
								dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1];
						else
							if (dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1] > dynamic_map_x_value[j*sprectx.at(x).width + i - 1])
								dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1];
							else
								dynamic_map_x_value[j*sprectx.at(x).width + i] = ptr_roi[j*sprectx.at(x).width + i] + dynamic_map_x_value[j*sprectx.at(x).width + i - 1];
					}
				}
			}
		}

		//from last col
		int j;
		for (j = 0; j < sprectx.at(x).height; j++) {
			if (dynamic_map_x_value[(sprectx.at(x).width - 1) + dynamic_seam_x.at(sprectx.at(x).width - 1)*sprectx.at(x).width] < dynamic_map_x_value[(sprectx.at(x).width - 1) + j*sprectx.at(x).width]) {
				//if(ptrmask[j*roirect.at(x).width+ roirect.at(x).width]==255)
				dynamic_seam_x.at(sprectx.at(x).width - 1) = j;
				//cout << "Dynamic_seam_x.at(roirect.at(x).width - 1)" << Dynamic_seam_x.at(roirect.at(x).width - 1) << endl;
			}
		}
		for (int i = sprectx.at(x).width - 2; i >= 0; i--) {
			//cout << "width i:" << i << endl;
			j = dynamic_seam_x.at(i + 1);
			//cout << "height j:" << j << endl;
			//top boundary
			if (j == 0) {
				if (dynamic_map_x_value[j*sprectx.at(x).width + i] <= dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i])
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					dynamic_seam_x.at(i) = j + 1;
				//else continue;
				else
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					dynamic_seam_x.at(i) = j;
				//else continue;
			}
			//down boundary
			else if (j == sprectx.at(x).height - 1) {
				if (dynamic_map_x_value[j*sprectx.at(x).width + i] <= dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i])
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					dynamic_seam_x.at(i) = j - 1;
				//else continue;
				else
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					dynamic_seam_x.at(i) = j;
				//else continue;
			}
			//middle
			//if((j!= sprectx.at(x).width - 1)&&(j!=0)) {
			else {
				if (dynamic_map_x_value[j*sprectx.at(x).width + i] < dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i]) {
					if (dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i] > dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i])
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						dynamic_seam_x.at(i) = j - 1;
					//else continue;
					else
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						dynamic_seam_x.at(i) = j + 1;
					//else continue;
				}
				else {
					if (dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i] > dynamic_map_x_value[j*sprectx.at(x).width + i])
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						dynamic_seam_x.at(i) = j - 1;
					//else continue;
					else
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						dynamic_seam_x.at(i) = j;
					//else continue;
				}
			}

		}
		//cout << "x :" << x << endl;
		dynamic_seam_sp.at(x + src.cols / SP) = (dynamic_seam_x);
		dynamic_seam_x.clear();
		dynamic_map_x_value.clear();
		

	}
	//********************************************************************************

	//initial rst 
	metaout.resize((src.cols)*(src.rows));
	unsigned char *rgb = (unsigned char*)(src.data);

	rst.resize((src.cols / SP + 1)*(src.rows / SP + 1));
	int block_x_cnt = 0;
	int block_y_cnt = 0;
	
	for (int i = 0; i < src.rows; i++){
		block_y_cnt = 0;
		for (int j = 0; j < src.cols; j++){
			if (block_y_cnt != src.cols / SP){
				if (j > dynamic_seam_sp.at(block_y_cnt).at(i) + sprecty.at(block_y_cnt).x){
					if (block_y_cnt < src.cols / SP)
						block_y_cnt = block_y_cnt + 1;
				}				
			}

			if (block_x_cnt == src.rows / SP){
				if (i < dynamic_seam_sp.at((src.cols / SP) + block_x_cnt - 1).at(j) + sprectx.at(block_x_cnt - 1).y)
					block_x_cnt--;
			}
			else if (block_x_cnt == 0){
				if (i >= dynamic_seam_sp.at((src.cols / SP) + block_x_cnt).at(j) + sprectx.at(block_x_cnt).y)
					block_x_cnt = block_x_cnt + 1;
			}
			else{
				if (i >= dynamic_seam_sp.at((src.cols / SP) + block_x_cnt).at(j) + sprectx.at(block_x_cnt).y)
					block_x_cnt = block_x_cnt + 1;
				else if (i < dynamic_seam_sp.at((src.cols / SP) + block_x_cnt - 1).at(j) + sprectx.at(block_x_cnt - 1).y)
					block_x_cnt = block_x_cnt - 1;
				else
					block_x_cnt = block_x_cnt;
			}

			rst.at(block_y_cnt + block_x_cnt*((src.cols / SP) + 1)).push_back(i*src.cols + j);
			metaout.at(i*src.cols + j).color = src.at<Vec3b>(i, j);
			metaout.at(i*src.cols + j).spnum = (block_y_cnt + block_x_cnt*((src.cols / SP) + 1));
			avgcolor[metaout.at(i*src.cols + j).spnum].r = avgcolor[metaout.at(i*src.cols + j).spnum].r + (int)src.at<Vec3b>(i, j)[2];
			avgcolor[metaout.at(i*src.cols + j).spnum].g = avgcolor[metaout.at(i*src.cols + j).spnum].g + (int)src.at<Vec3b>(i, j)[1];
			avgcolor[metaout.at(i*src.cols + j).spnum].b = avgcolor[metaout.at(i*src.cols + j).spnum].b + (int)src.at<Vec3b>(i, j)[0];
		}
	}

	for (int i = 0; i < rst.size(); i++) {
		avgcolor[i].cnt = rst.at(i).size();
		if (avgcolor[i].cnt == 0) {
			continue;
		}
		else {
			avgcolor[i].r = avgcolor[i].r / avgcolor[i].cnt;
			avgcolor[i].g = avgcolor[i].g / avgcolor[i].cnt;
			avgcolor[i].b = avgcolor[i].b / avgcolor[i].cnt;
		}
		supout.at(i).avgcolor = Vec3b(avgcolor[i].b, avgcolor[i].g, avgcolor[i].r);
		supout.at(i).label = i;
		supout.at(i).size = rst.at(i).size();
		supout.at(i).flag = true;
		supout.at(i).layer = 1;

	}
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			metaout.at(i*src.cols + j).sim = simil(Vec3b(avgcolor[metaout.at(i*src.cols + j).spnum].b, avgcolor[metaout.at(i*src.cols + j).spnum].g, avgcolor[metaout.at(i*src.cols + j).spnum].r), src.at<Vec3b>(i, j));
		}
	}

	//******avg simility
	double simavg;
	for (int x = 0; x < rst.size(); x++) {
		simavg = 0;
		for (int i = 0; i < supout.at(x).size; i++) {
			simavg = simavg + metaout.at(rst.at(x).at(i)).sim;
		}
		supout.at(x).var = simavg;// / supout.at(x).size;
	}
	//******avg simility

	//******simily to img
	Mat simility;
	simility = Mat::zeros(src.size(), CV_32F);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			simility.at<float>(i, j) = metaout.at(i*src.cols + j).sim;
		}
	}

	imshow("simility", simility);
	Mat similityblock;
	similityblock = Mat::zeros(src.size(), CV_32F);
	int maxtemp = 0;
	for (int x = 0; x < rst.size(); x++) {
		if (maxtemp <= supout.at(x).var)
			maxtemp = supout.at(x).var;
		else
			maxtemp = maxtemp;
	}
	for (int x = 0; x < rst.size(); x++) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (x == metaout.at(i*src.cols + j).spnum)
					similityblock.at<float>(i, j) = supout.at(x).var / maxtemp;
			}
		}
	}
	//normalize(similityblock, similityblock, 1,0,NORM_MINMAX );
	imshow("similityblock", similityblock);
	//*****simily to img

	avgcolor.clear();
	avgcolor.shrink_to_fit();


	cout << "xxxxx" << endl;
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


double simil(Vec3b a, Vec3b b) {
	//*********version 1 simility from reference 
	/*
	double s;
	double v1, v2, v3;

	if (a == b)
	s = 1;
	else {
	v1 = abs(a[0] - b[0]);
	v2 = abs(a[1] - b[1]);
	v3 = abs(a[2] - b[2]);
	s = 9 / ((v1 + v2 + v3) * (1 / v1 + 1 / v2 + 1 / v3));
	}
	return s;
	*/



	//*********version 2 Euclidean RGB distance 

	double s;
	double v1, v2, v3;
	double d;
	v1 = abs((double)a[0] - (double)b[0]) / 255;
	v2 = abs((double)a[1] - (double)b[1]) / 255;
	v3 = abs((double)a[2] - (double)b[2]) / 255;
	s = v1 + v2 + v3 / 3;

	return s;

}

void iteratorseam(vector <vector<int>> &superpixels, Mat &energyx, Mat &energyy, vector <vector<int>> &rst, vector <meta> &metaout, vector <superpixel> &supout) {
	vector <Rect> sprect;
	vector <vector <int>*> colwidthvec;
	vector <int> rowvalue;



	//cout << superpixels.size();
	int rowindex;
	int colindex;
	int rowwidth;
	int colwidth;
	for (int i = 0; i < superpixels.size(); i++) {
		if (superpixels.at(i).size() == 0) {
			//continue;
			Rect spmat(0, 0, 0, 0);
			sprect.push_back(spmat);
		}
		else {
			rowindex = superpixels.at(i).at(0) / src.cols;
			//colindex = superpixels.at(i).at(0) % src.cols; //has problem
			//cout << rowindex << " " << colindex << endl;
			//cout << "superpixels.at(i).size()  :" << superpixels.at(i).size()  << endl;
			rowwidth = ((superpixels.at(i).at(superpixels.at(i).size() - 1)) / src.cols) + 1;
			//cout << "rowwidth  "<<rowwidth << endl;
			for (int m = 0; m < rowwidth; m++) {
				colwidthvec.push_back(new vector<int>);
			}
			//cout << "superpixels.at(i).size()" << superpixels.at(i).size() << endl;
			int temprowwidth = 0;
			int temprowwidthmin = src.cols;
			int temprowwidthmax = 0;
			int tempcolindex = src.cols;
			for (int j = 0; j < superpixels.at(i).size(); j++) {
				colwidthvec.at(superpixels.at(i).at(j) / src.cols)->push_back((int)superpixels.at(i).at(j));
				//cout << "superpixels.at(i).at(j) : " << superpixels.at(i).at(j) << endl;
			}
			for (int n = 0; n < colwidthvec.size(); n++) {

				for (int x = 0; x < colwidthvec.at(n)->size(); x++) {
					if (temprowwidthmax <= colwidthvec.at(n)->at(x) % src.cols)
						temprowwidthmax = colwidthvec.at(n)->at(x) % src.cols;
					else
						temprowwidthmax = temprowwidthmax;
					if (temprowwidthmin >= colwidthvec.at(n)->at(x) % src.cols)
						temprowwidthmin = colwidthvec.at(n)->at(x) % src.cols;
					else
						temprowwidthmin = temprowwidthmin;
				}

				if (colwidthvec.at(n)->size() == 0)
					continue;
				else {
					if (tempcolindex >= colwidthvec.at(n)->at(0) % src.cols)
						tempcolindex = colwidthvec.at(n)->at(0) % src.cols;
					else
						tempcolindex = tempcolindex;
				}
				temprowwidth = temprowwidthmax - temprowwidthmin + 1;
			}
			colwidth = temprowwidth;
			colindex = tempcolindex;
			rowwidth = rowwidth - rowindex;
			//cout << " xxx :" << i << endl;
			//cout << "colindex :" << colindex << "rowindex :" << rowindex << "colwidth :" << colwidth << "rowwidth :" << rowwidth << endl;
			Rect spmat(colindex, rowindex, colwidth, rowwidth);
			//Mat sproi;
			//energyx(spmat).copyTo(sproi);
			//imshow("roi", sproi);
			sprect.push_back(spmat);
			supout.at(i).rectangle = spmat; //rectangle not roi
			colwidthvec.clear();
		}
	}
	//vector <vector<int>> rst;


	superpixelsplit(superpixels, sprect, energyy, energyx, rst, metaout, supout);
	//	imshow("energyx", energyx);
	//	imshow("energyy", energyy);

}

void superpixelsplit(vector<vector<int>> sp, vector <Rect> &roirect, Mat & energy_y, Mat & energy_x, vector <vector<int>> &rst, vector <meta> &meta, vector <superpixel> &sup) {

	Mat roi;
	Mat mask;
	Mat weimap;
	uchar *ptrmask;
	float *ptrroi;

	int col, row;
	vector <float> weightmap_y;
	vector <float> weightmap_x;
	vector <float> Dynamic_map_y_value;
	vector <int> Dynamic_seam_y;
	vector <float> Dynamic_map_x_value;
	vector <int> Dynamic_seam_x;
	vector <vector<int>> Dynamic_seam_sp;


	vector<avg_color_> avgcolor;
	avgcolor.reserve(5000);
	avgcolor.resize(5000);

	Dynamic_seam_sp.resize(2 * sp.size());

	//********************************y_axis********************************************************
	for (int x = 0; x < roirect.size(); x++) {
		double center = gaussian(0, roirect.at(x).width);
		weightmap_y.resize(roirect.at(x).width);
		for (int i = -roirect.at(x).width / 2; i < roirect.at(x).width / 2; i++)
			weightmap_y.at(roirect.at(x).width / 2 + i) = gaussian(i, roirect.at(x).width) / center;


		if (sp.at(x).size() == 0) {
			continue;
		}
		else {
			energy_y(roirect[x]).copyTo(roi);
			mask = Mat::zeros(roi.size(), CV_8UC1);

			for (int y = 0; y < sp.at(x).size(); y++) {
				col = (sp.at(x).at(y) % src.cols) - roirect[x].x;
				row = (sp.at(x).at(y) / src.cols) - roirect[x].y;
				mask.at<uchar>(row, col) = 255;
			}
			roi.convertTo(roi, CV_32F);
			ptrmask = mask.data;
			ptrroi = roi.ptr<float>();
		}
		/*	Mat roivalue;
		roivalue = mask&roi;
		roivalue.convertTo(roivalue, CV_32F);
		sup.at(x).roiyvalue = roivalue;
		roi.convertTo(roi, CV_32F);
		sup.at(x).mask = mask;*/


		Dynamic_map_y_value.resize(roirect.at(x).height*roirect.at(x).width);
		Dynamic_seam_y.resize(roirect.at(x).height);
		for (int i = 0; i < roirect.at(x).height; i++) {
			for (int j = 0; j < roirect.at(x).width; j++) {
				ptrroi[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] * weightmap_y[j];// *weightmap_y[j];
			}
		}

		for (int i = 0; i < roirect.at(x).height; i++) {
			for (int j = 0; j < roirect.at(x).width; j++) {

				//First row value from weight roi
				if (i == 0) {
					Dynamic_map_y_value[j] = ptrroi[j];
					//cout << "Dynamic_map_y_value[j]" << Dynamic_map_y_value[j] << endl;
				}
				//Following rows find the largest from above connected to itself
				else {
					if (j == 0) {	//left boundary
						if (Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j] > Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1])
							Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j];
						else
							Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1];
					}
					if (j == roirect.at(x).width) { //right  boundary
						if (Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j - 1] > Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j])
							Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j - 1];
						else
							Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j];
					}
					else {	// middle 
						if (Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j - 1] > Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j])
							if (Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j - 1] > Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1])
								Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j - 1];
							else
								Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1];
						else
							if (Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j] > Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1])
								Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j];
							else
								Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1];
					}
				}
			}
		}
		/*weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
		//*****************show the weight map*********************
		for (int y = 0; y < Dynamic_map_y_value.size(); y++) {

		//weimap.create(mask.size(), CV_32F);
		//weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
		float * ptrweimap = weimap.ptr<float>(0);
		ptrweimap[y] = Dynamic_map_y_value[y];
		//imshow("weimap", weimap);
		}
		//*****************show the weight map*********************
		*/
		if ((roirect.at(x).width < 10) || (roirect.at(x).width == 0) || ((sup.at(x).var<0.5) && (sup.at(x).size < 500))) {	//condition to find seams
			Dynamic_seam_sp.at(x) = (Dynamic_seam_y);
			//cout << "pushback null y DSY " << roirect.at(x).width << endl;
		}
		else {
			//from last row
			int j;
			for (j = 0; j < roirect.at(x).width; j++) {				//*************************-1???************************************
				if (Dynamic_map_y_value[(roirect.at(x).height - 1)*roirect.at(x).width + Dynamic_seam_y.at(roirect.at(x).height - 1)] < Dynamic_map_y_value[(roirect.at(x).height - 1)*roirect.at(x).width + j]) {
					//if (ptrmask[(roirect.at(x).height - 1)*roirect.at(x).width + j] == 255) {
					Dynamic_seam_y.at(roirect.at(x).height - 1) = j;
					//cout <<"last row j : " <<j << endl;
					//}
					//cout << "Dynamic_seam_y.at(roirect.at(x).height - 1)" << Dynamic_seam_y.at(roirect.at(x).height - 1) << endl;
				}
			}
			for (int i = roirect.at(x).height - 2; i >= 0; i--) {
				//cout << "i:" << i << endl;
				j = Dynamic_seam_y.at(i + 1);
				//cout << "j:" << j << endl;
				//left boundary
				if (j == 0) {
					if (Dynamic_map_y_value[i*roirect.at(x).width + j] <= Dynamic_map_y_value[i*roirect.at(x).width + j + 1]) {
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						Dynamic_seam_y.at(i) = j + 1;
						//else
						//continue;
					}
					else {
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						Dynamic_seam_y.at(i) = j;
						//else
						//continue;
					}
				}
				//right boundary
				else if (j == roirect.at(x).width - 1) {
					if (Dynamic_map_y_value[i*roirect.at(x).width + j] <= Dynamic_map_y_value[i*roirect.at(x).width + j - 1]){
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						Dynamic_seam_y.at(i) = j - 1;
						//else continue;
					}
					else {
						//if (ptrmask[i*roirect.at(x).width + j] == 255)
						Dynamic_seam_y.at(i) = j;
						//else continue;
					}
				}
				//middle
				//if((j!= roirect.at(x).width - 1)&&(j!=0)) {
				else {
					if (Dynamic_map_y_value[i*roirect.at(x).width + j] < Dynamic_map_y_value[i*roirect.at(x).width + j + 1]) {
						if (Dynamic_map_y_value[i*roirect.at(x).width + j - 1] > Dynamic_map_y_value[i*roirect.at(x).width + j + 1]) {
							//if (ptrmask[i*roirect.at(x).width + j] == 255)
							Dynamic_seam_y.at(i) = j - 1;
							//else continue;
						}
						else {
							//if (ptrmask[i*roirect.at(x).width + j] == 255)
							Dynamic_seam_y.at(i) = j + 1;
							//else continue;
						}
					}
					else {
						if (Dynamic_map_y_value[i*roirect.at(x).width + j - 1] > Dynamic_map_y_value[i*roirect.at(x).width + j]) {
							//if (ptrmask[i*roirect.at(x).width + j] == 255)
							Dynamic_seam_y.at(i) = j - 1;
							//else continue;
						}
						else {
							//if (ptrmask[i*roirect.at(x).width + j] == 255)
							Dynamic_seam_y.at(i) = j;
							//else continue;
						}
					}
				}

			}
			//cout << "x :" << x << endl;
			Dynamic_seam_sp.at(x) = (Dynamic_seam_y);
			Dynamic_seam_y.clear();
			Dynamic_map_y_value.clear();
		}
	}
	//********************************y_axis********************************************************

	//********************************x_axis********************************************************
	for (int x = 0; x < roirect.size(); x++) {
		double center = gaussian(0, roirect.at(x).height);
		weightmap_x.resize(roirect.at(x).height);
		for (int i = -roirect.at(x).height / 2; i < roirect.at(x).height / 2; i++)
			weightmap_x.at(roirect.at(x).height / 2 + i) = gaussian(i, roirect.at(x).height) / center;

		if (sp.at(x).size() == 0) {
			continue;
		}
		else {
			energy_x(roirect[x]).copyTo(roi);
			mask = Mat::zeros(roi.size(), CV_8UC1);

			for (int y = 0; y < sp.at(x).size(); y++) {
				col = (sp.at(x).at(y) % src.cols) - roirect[x].x;
				row = (sp.at(x).at(y) / src.cols) - roirect[x].y;
				mask.at<uchar>(row, col) = 255;
			}
			roi.convertTo(roi, CV_32F);
			ptrmask = mask.data;
			ptrroi = roi.ptr<float>();
		}
		/*Mat roivalue;
		roivalue = mask&roi;
		roivalue.convertTo(roivalue, CV_32F);
		sup.at(x).roixvalue = roivalue;
		roi.convertTo(roi, CV_32F);
		sup.at(x).mask = mask;
		*/
		Dynamic_map_x_value.resize(roirect.at(x).height*roirect.at(x).width);
		Dynamic_seam_x.resize(roirect.at(x).width);
		for (int i = 0; i < roirect.at(x).height; i++) {
			for (int j = 0; j < roirect.at(x).width; j++) {
				ptrroi[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] * weightmap_x[i];// *weightmap_x[i];
			}
		}

		for (int i = 0; i < roirect.at(x).width; i++) {
			for (int j = 0; j < roirect.at(x).height; j++) {

				//First col value from weight roi
				if (i == 0) {

					Dynamic_map_x_value[j*roirect.at(x).width] = ptrroi[j*roirect.at(x).width];
					//cout << "Dynamic_map_x_value[j*roirect.at(x).width]" << Dynamic_map_x_value[j*roirect.at(x).width] << endl;

				}
				//Following cols find the largest from above connected to itself
				else {
					if (j == 0) {	//top boundary
						if (Dynamic_map_x_value[j*roirect.at(x).width + i - 1] > Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1])
							Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[j*roirect.at(x).width + i - 1];
						else
							Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1];
					}
					else if (j == roirect.at(x).height - 1) { //down  boundary
						if (Dynamic_map_x_value[j*roirect.at(x).width + i - 1] > Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1])
							Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[j*roirect.at(x).width + i - 1];
						else
							Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1];
					}
					else {	// middle 
						if (Dynamic_map_x_value[j*roirect.at(x).width + i - 1] < Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1])
							if (Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1] < Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1])
								Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1];
							else
								Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1];
						else
							if (Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1] > Dynamic_map_x_value[j*roirect.at(x).width + i - 1])
								Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1];
							else
								Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[j*roirect.at(x).width + i - 1];
					}
				}
			}
		}
		/*weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
		//*****************show the weight map*********************
		for (int y = 0; y < Dynamic_map_x_value.size(); y++) {

		//weimap.create(mask.size(), CV_32F);
		//weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
		float * ptrweimap = weimap.ptr<float>(0);
		ptrweimap[y] = Dynamic_map_x_value[y];
		//imshow("weimap", weimap);
		}
		//*****************show the weight map*********************
		*/
		if (roirect.at(x).height < 10 || (roirect.at(x).height == 0) || ((sup.at(x).var<0.5) && (sup.at(x).size <500))) {	//condition to find seams
			Dynamic_seam_sp.at(x + sp.size()) = (Dynamic_seam_x);
			//cout << "pushback null DSX  height: "<< roirect.at(x).height << endl;
		}
		else {
			//from last col
			int j;
			for (j = 0; j < roirect.at(x).height; j++) {
				if (Dynamic_map_x_value[(roirect.at(x).width - 1) + Dynamic_seam_x.at(roirect.at(x).width - 1)*roirect.at(x).width] < Dynamic_map_x_value[(roirect.at(x).width - 1) + j*roirect.at(x).width]) {
					//if(ptrmask[j*roirect.at(x).width+ roirect.at(x).width]==255)
					Dynamic_seam_x.at(roirect.at(x).width - 1) = j;
					//cout << "Dynamic_seam_x.at(roirect.at(x).width - 1)" << Dynamic_seam_x.at(roirect.at(x).width - 1) << endl;
				}
			}
			for (int i = roirect.at(x).width - 2; i >= 0; i--) {
				//cout << "width i:" << i << endl;
				j = Dynamic_seam_x.at(i + 1);
				//cout << "height j:" << j << endl;
				//top boundary
				if (j == 0) {
					if (Dynamic_map_x_value[j*roirect.at(x).width + i] <= Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i])
						//if (ptrmask[j*roirect.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j + 1;
					//else continue;
					else
						//if (ptrmask[j*roirect.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j;
					//else continue;
				}
				//down boundary
				else if (j == roirect.at(x).height - 1) {
					if (Dynamic_map_x_value[j*roirect.at(x).width + i] <= Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i])
						//if (ptrmask[j*roirect.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j - 1;
					//else continue;
					else
						//if (ptrmask[j*roirect.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j;
					//else continue;
				}
				//middle
				//if((j!= roirect.at(x).width - 1)&&(j!=0)) {
				else {
					if (Dynamic_map_x_value[j*roirect.at(x).width + i] < Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i]) {
						if (Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i] > Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i])
							//if (ptrmask[j*roirect.at(x).width + i] == 255)
							Dynamic_seam_x.at(i) = j - 1;
						//else continue;
						else
							//if (ptrmask[j*roirect.at(x).width + i] == 255)
							Dynamic_seam_x.at(i) = j + 1;
						//else continue;
					}
					else {
						if (Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i] > Dynamic_map_x_value[j*roirect.at(x).width + i])
							//if (ptrmask[j*roirect.at(x).width + i] == 255)
							Dynamic_seam_x.at(i) = j - 1;
						//else continue;
						else
							//if (ptrmask[j*roirect.at(x).width + i] == 255)
							Dynamic_seam_x.at(i) = j;
						//else continue;
					}
				}

			}
			//cout << "x :" << x << endl;
			Dynamic_seam_sp.at(x + sp.size()) = (Dynamic_seam_x);
			Dynamic_seam_x.clear();
			Dynamic_map_x_value.clear();
		}
	}

	//********************************x_axis********************************************************

	clock_t inner = clock();




	//initial rst
	rst.resize(4 * sp.size());

	for (int x = 0; x < sp.size(); x++) {
		for (int y = 0; y < sp.at(x).size(); y++) {
			//cout << "x :" << x << "  y :" << y << endl;
			col = (sp.at(x).at(y) % src.cols) - roirect[x].x;
			row = (sp.at(x).at(y) / src.cols) - roirect[x].y;
			if ((col >= Dynamic_seam_sp.at(x).at(row)) && (row >= Dynamic_seam_sp.at(sp.size() + x).at(col))) {
				//cout << "3  " << ((col > Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
				rst.at(4 * x + 3).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).color = src.at<Vec3b>(row + roirect[x].y, row + roirect[x].x);
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum = (4 * x + 3);
			}
			else if ((col < Dynamic_seam_sp.at(x).at(row)) && (row < Dynamic_seam_sp.at(sp.size() + x).at(col))) {
				//cout << "0  " << ((col <= Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
				rst.at(4 * x).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).color = src.at<Vec3b>(row + roirect[x].y, row + roirect[x].x);
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum = (4 * x);

			}
			else if ((col >= Dynamic_seam_sp.at(x).at(row)) && (row < Dynamic_seam_sp.at(sp.size() + x).at(col))) {
				//cout << "1  " << ((col > Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
				rst.at(4 * x + 1).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).color = src.at<Vec3b>(row + roirect[x].y, row + roirect[x].x);
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum = (4 * x + 1);
			}
			else if ((col < Dynamic_seam_sp.at(x).at(row)) && (row >= Dynamic_seam_sp.at(sp.size() + x).at(col))) {
				//cout << "2  " << ((col <= Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
				rst.at(4 * x + 2).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).color = src.at<Vec3b>(row + roirect[x].y, row + roirect[x].x);
				meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum = (4 * x + 2);
			}

			avgcolor[meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum].r = avgcolor[meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum].r + (int)src.at<Vec3b>(roirect[x].y, roirect[x].x)[2];
			avgcolor[meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum].g = avgcolor[meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum].g + (int)src.at<Vec3b>(roirect[x].y, roirect[x].x)[1];
			avgcolor[meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum].b = avgcolor[meta.at((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x)).spnum].b + (int)src.at<Vec3b>(roirect[x].y, roirect[x].x)[0];
		}
	}
	for (int i = 0; i < rst.size(); i++) {
		avgcolor[i].cnt = rst.at(i).size();
		if (avgcolor[i].cnt == 0) {
			continue;
		}
		else {
			avgcolor[i].r = avgcolor[i].r / avgcolor[i].cnt;
			avgcolor[i].g = avgcolor[i].g / avgcolor[i].cnt;
			avgcolor[i].b = avgcolor[i].b / avgcolor[i].cnt;
		}
		sup.at(i).avgcolor = Vec3b(avgcolor[i].b, avgcolor[i].g, avgcolor[i].r);
		sup.at(i).label = i;
		sup.at(i).size = rst.at(i).size();
		sup.at(i).flag = true;
		sup.at(i).layer = 2;
	}
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			meta.at(i*src.cols + j).sim = simil(Vec3b(avgcolor[meta.at(i*src.cols + j).spnum].b, avgcolor[meta.at(i*src.cols + j).spnum].g, avgcolor[meta.at(i*src.cols + j).spnum].r), src.at<Vec3b>(i, j));
		}
	}

	//******avg simility
	double simavg;
	for (int x = 0; x < rst.size(); x++) {
		simavg = 0;
		for (int i = 0; i < rst.at(x).size(); i++) {
			simavg = simavg + meta.at(rst.at(x).at(i)).sim;
		}
		sup.at(x).var = simavg;// / supout.at(x).size;
	}
	//******avg simility

	//******simily to img
	Mat simility;
	simility = Mat::zeros(src.size(), CV_32F);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			simility.at<float>(i, j) = meta.at(i*src.cols + j).sim;
		}
	}

	imshow("simility2", simility);
	Mat similityblock;
	similityblock = Mat::zeros(src.size(), CV_32F);
	int maxtemp = 0;
	for (int x = 0; x < rst.size(); x++) {
		if (maxtemp <= sup.at(x).var)
			maxtemp = sup.at(x).var;
		else
			maxtemp = maxtemp;
	}
	for (int x = 0; x < rst.size(); x++) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (x == meta.at(i*src.cols + j).spnum)
					similityblock.at<float>(i, j) = sup.at(x).var / maxtemp;
			}
		}
	}

	clock_t innerend = clock();
	cout << "Time elapsed inner : " << (double)(innerend - inner) / CLOCKS_PER_SEC << endl;
	//normalize(similityblock, similityblock, 1,0,NORM_MINMAX );
	imshow("similityblock2", similityblock);
	//*****simily to img

}
//method 2
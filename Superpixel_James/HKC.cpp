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
//#include"ldp.h"

#define PI  3.14159265
#define DS 1			//colors (original is 256 (8 bits))
#define SP 50			//30 must smaller than [SRCSHORTSIDE/2] i.e 321/2 = 160
#define SPATIALWEIGHT 0
#define threshvalue 50
#define Iter 4

using namespace cv;
using namespace std;

Mat src;
Mat srcds;
Mat sobelrstx, sobelrsty;
Mat rst;
Mat dumplabel;
Mat seamx, seamy;

int glogalspcounter;
int imgcnt;
char filename_image[70];
char filename_label[80];
char resultfolder[70];
char resultname[70];

vector<int> label;

struct meta {
	Vec3b color;
	double sim;
	int spnum;
};

struct superpixel {
	Rect rectangle;
	Mat roiyvalue;
	Mat roixvalue;
	Mat mask;
	int label;
	Vec3b avgcolor;
	//double sim;	//similiarity should in meta
	int size;
	double var;
	int layer;
	bool flag;
};

void colordownsample(const Mat& src, Mat& rst, int dsrate);
double gaussian(int x, int y);
void energy(const Mat& src, Mat &rst);
void imgtovertex(Mat src);
void dumpsuperpixel(const Mat& seammapx, const Mat& seammapy, const Mat& src, vector <vector<int>> & superpixels);
void test(Mat& x, Mat& seamx, Mat& y, Mat& seamy);
void iteratorseam(vector <vector<int>> &superpixels, Mat &energyx, Mat &energyy, vector <vector<int>> &rst, vector <meta> &meta, vector <superpixel> &sup);
void firstseam(vector <vector<int>> &superpixels, Mat &energyx, Mat &energyy, vector <vector<int>> &rst, vector <meta> &metaout, vector <superpixel> &supout);
void superpixelsplit(vector<vector<int>> sp, vector <Rect> &roirect, Mat & energy_y, Mat & energy_x, vector <vector<int>> &rst, vector <meta> &meta, vector <superpixel> &sup);
void itersp(vector<vector<int>> sp, Mat & energy_y, Mat & energy_x, vector <vector<int>> &rst, vector <meta> &meta, vector <superpixel> &sup);
//void superpixelsplitx(vector<vector<int>> sp, vector <Rect> &roirect, Mat & energy_y, Mat & energy_x, vector <vector<int>> &rst);
//void superpixelsplity(vector<vector<int>> sp, vector <Rect> &roirect, Mat & energy_y, Mat & energy_x, vector <vector<int>> &rst);
void vec2txt(vector<vector<int>> src);

double simil(Vec3b a, Vec3b b);
/*
struct superpixel {
int label;
Vec3b avgcolor;
double similation;
};*/

struct avg_color_
{
	int r;
	int g;
	int b;
	int cnt;
};

struct LAB {
	double L;
	double A;
	double B;
};



uchar *src_labptr;

int labvarweight(int i, int j);

// LDP filter
Mat M0 = (Mat_<float>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
Mat M1 = (Mat_<float>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
Mat M2 = (Mat_<float>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
Mat M3 = (Mat_<float>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
Mat M4 = (Mat_<float>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
Mat M5 = (Mat_<float>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
Mat M6 = (Mat_<float>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);
Mat M7 = (Mat_<float>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);
//filter2D(src,dst,dst.depth(),M0,Point(-1,-1));
//LDP filter

int main(int argc, char **argv)
{
	//**********************************random color*************************//
	vector<Vec3b>random_color;
	for (int rr = 0; rr < 50000; rr++)
		random_color.push_back(Vec3b(rand() % 255, rand() % 255, rand() % 255));
	//**********************************random color*************************//
	sprintf(resultfolder, "hkc%d", SP);
	_mkdir(resultfolder);
	int tempspcnt = 0;
	for (imgcnt = 1; imgcnt <= 50; imgcnt++) {
		sprintf(filename_image, "data50/image (%d).jpg", imgcnt);
		sprintf(filename_label, "hkc%d/label (%d).txt", SP, imgcnt);
		src = imread(filename_image, 1);//153  47bird  4??!!
		imshow("src", src);
		cout << filename_image << "load" << endl;
		//******************************
		vector <vector<int>> superpixels;
		vector <vector<int>> rst;

		clock_t startlab = clock();
		Mat src_lab;
		src_lab = src.clone();
		cvtColor(src_lab, src_lab, CV_BGR2Lab);
		clock_t endslab = clock();
		cout << "Time elapsed LAB : " << (double)(endslab - startlab) / CLOCKS_PER_SEC << endl;
		src_labptr = src_lab.data;
		//int test;
		//test = labvarweight(1, 2);
		//cout << test << endl;


		//******************************

		//resize(src, src, Size(src.cols / 2, src.rows / 2));


		clock_t startk = clock();

		//*****k means******************
		Mat samples(src.cols*src.rows, 3, CV_32F);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				for (int z = 0; z < 3; z++)
					samples.at< float>(i + j*src.rows, z) = src.at< Vec3b>(i, j)[z];
		int clusterCount = 8; // 分八種?
		Mat klabels;
		int attempts = 10; //執行10次
		Mat centers;
		kmeans(samples, clusterCount, klabels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 1.0), attempts, KMEANS_RANDOM_CENTERS, centers);
		Mat kimage(src.size(), src.type());
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				int cluster_idx = klabels.at< int>(i + j*src.rows, 0);
				kimage.at< Vec3b>(i, j)[0] = centers.at< float>(cluster_idx, 0);
				kimage.at< Vec3b>(i, j)[1] = centers.at< float>(cluster_idx, 1);
				kimage.at< Vec3b>(i, j)[2] = centers.at< float>(cluster_idx, 2);
			}
		}

		cvtColor(kimage, kimage, CV_BGR2GRAY);


		//*****k means******************
		;
		clock_t endsk = clock();
		//cout << "Time elapsed kimg : " << (double)(endsk - startk) / CLOCKS_PER_SEC << endl;

		//**********************************color down sample*******************
		//colordownsample(src, srcds, DS);
		//imshow("srcde", srcds);
		//**********************************color down sample*******************



		//**********************************Soble operation**********************
		Mat srcB;
		srcB = src.clone();
		cvtColor(srcB, srcB, CV_BGR2GRAY);


		////kirsch filter
		//Mat a, b, c, d, e, f, g, h;
		//Mat kiredge;
		//filter2D(srcB, a, a.depth(), M0, Point(-1, -1));
		//filter2D(srcB, b, b.depth(), M1, Point(-1, -1));
		//filter2D(srcB, c, c.depth(), M2, Point(-1, -1));
		//filter2D(srcB, d, d.depth(), M3, Point(-1, -1));
		//filter2D(srcB, e, e.depth(), M4, Point(-1, -1));
		//filter2D(srcB, f, f.depth(), M5, Point(-1, -1));
		//filter2D(srcB, g, g.depth(), M6, Point(-1, -1));
		//filter2D(srcB, h, h.depth(), M7, Point(-1, -1));

		//kiredge = a / 8 + b / 8 + c / 8 + d / 8 + e / 8 + f / 8 + g / 8 + h / 8;

		//imshow("a", a);
		//imshow("b", b);
		//imshow("c", c);
		//imshow("d", d);
		//imshow("e", e);
		//imshow("f", f);
		//imshow("g", g);
		//imshow("h", h);
		//imshow("kiredge", kiredge);
		////kirsch filter
		//normalize(kiredge, kiredge, 0, 255,CV_MINMAX);
		/*
		LDP ldp;
		Mat ldpimg;
		vector <LDP::PAIR> ldpweight;
		vector<vector<vector<int>>> ldpweight3d(src.rows, vector<vector<int>>(src.cols, vector<int>(8)));
		vector<vector<vector<int>>> ldpweight3dtrans(src.cols, vector<vector<int>>(src.rows, vector<int>(8)));


		ldp.ldppattern(srcB, ldpimg, 1, ldpweight);
		//imshow("ldpimg", ldpimg);
		//ldpweight transpose

		for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		for (int x = 0; x < 8; x++) {
		ldpweight3d[i][j][x] = ldpweight[(i*src.cols + j) * 8 + x].point_value;
		//cout << ldpweight3d[i][j][x] << endl;
		}

		for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		for (int x = 0; x < 8; x++) {
		ldpweight3dtrans[j][src.rows - 1 - i][(x + 2) % 8] = ldpweight3d[i][j][x];
		//cout << j << "," << src.rows - 1 - i << endl;
		//cout << ldpweight3dtrans[j][src.rows - 1 - i][(x + 2) % 8] << endl;
		}
		*/

		Mat lab[3];
		split(src_lab, lab);


		Mat clustersobelx, clustersobely;
		clock_t startedge = clock();
		Mat cannyedge;
		GaussianBlur(src_lab, src_lab, Size(3, 3), 0, 0);
		Canny(src_lab, cannyedge, 50, 250, 3, false);
		imshow("Canny", cannyedge);
		Sobel(src_lab, sobelrstx, CV_32F, 0, 1, 3, 1);
		Sobel(src_lab, sobelrsty, CV_32F, 1, 0, 3, 1);
		Mat sobel_lab_x[3];
		Mat sobel_lab_y[3];
		convertScaleAbs(sobelrstx, sobelrstx);
		convertScaleAbs(sobelrsty, sobelrsty);
		split(sobelrstx, sobel_lab_x);
		split(sobelrsty, sobel_lab_y);


		clock_t endsedge = clock();
		cout << "Time elapsed edgeenergy : " << (double)(endsedge - startedge) / CLOCKS_PER_SEC << endl;

		GaussianBlur(kimage, kimage, Size(3, 3), 0, 0);



		Sobel(kimage, clustersobelx, CV_32F, 0, 1, 3, 1);
		Sobel(kimage, clustersobely, CV_32F, 1, 0, 3, 1);
		//Canny(new_image, sobelrstx, 1, 3, 3);
		//Canny(new_image, sobelrsty, 1, 3, 3);
		//convertScaleAbs(sobelrstx, sobelrstx);// abs  negative to postive
		//convertScaleAbs(sobelrsty, sobelrsty);// abs  negative to postive

		imshow("sobelx", sobelrstx);
		imshow("sobely", sobelrsty);
		Mat energy_y, energy_x;
		energy_y.create(src.size(), CV_32F);
		energy_x.create(src.size(), CV_32F);;
		float *ptr_energy_y = energy_y.ptr<float>();
		float *ptr_energy_x = energy_x.ptr<float>();
		uchar *ptr_sobel_y = sobelrsty.ptr<uchar>();
		uchar *ptr_sobel_x = sobelrstx.ptr<uchar>();
		uchar *ptr_canny = cannyedge.ptr<uchar>();

		for (int i = 0; i < src.cols*src.rows; i++) {
			ptr_energy_y[i] = ((float)abs(ptr_sobel_y[i * 3]) + (float)abs(ptr_sobel_y[i * 3 + 1]) + (float)abs(ptr_sobel_y[i * 3 + 2]) + 0.25 * (float)ptr_canny[i]);
			ptr_energy_x[i] = ((float)abs(ptr_sobel_x[i * 3]) + (float)abs(ptr_sobel_x[i * 3 + 1]) + (float)abs(ptr_sobel_x[i * 3 + 2]) + 0.25 * (float)ptr_canny[i]);
		}

		//convertScaleAbs(energy_y, energy_y);// abs  negative to postive
		//convertScaleAbs(energy_x, energy_x);// abs  negative to postive


		Mat sobelrst;
		convertScaleAbs(clustersobelx, clustersobelx);// liner transfer to 8 bit unsign int
		convertScaleAbs(clustersobely, clustersobely);// liner transfer to 8 bit unsign int
		//addWeighted(sobelrstx, 0.5, sobelrsty, 0.5, 0, sobelrst);
		//addWeighted(energy_y, 1, cannyedge, 0.25, 0, energy_y);
		//addWeighted(energy_x, 1, cannyedge, 0.25, 0, energy_x);
		imshow("sobelx+canny", energy_x);
		imshow("sobely+canny", energy_y);
		//addWeighted(sobelrst, 0.5, clustersobelx, 0.5, 0, sobelrstx);
		//addWeighted(sobelrst, 0.5, clustersobely, 0.5, 0, sobelrsty);
		//sobelrst = sobelrstx ^ sobelrsty;
		//imshow("sobelrst", sobelrst);
		Mat sobelrsttrans;
		transpose(sobelrst, sobelrsttrans);
		flip(sobelrsttrans, sobelrsttrans, 3);
		//imshow("sobelrsttrans", sobelrsttrans);

		//**********************************Soble operation**********************//




		//**********************************HSV Value****************************//

		//vector<Vec3f> hsv;
		//vector<Vec2f> hxy;
		//for(int i=0;i<src.rows;i++)
		//	for (int j = 0; j < src.cols; j=j+3) {
		//		float h, s, v;
		//		float r, g, b;
		//		b = (src.at<unsigned char>(i*j + j)) / 255.0;
		//		g = (src.at<unsigned char>(i*j + j + 1)) / 255.0;
		//		r = (src.at<unsigned char>(i*j + j + 2)) / 255.0;
		//		//cout << (int)src.at<unsigned char>(i*j + j) << "  ,  " << (int)src.at<unsigned char>(i*j + j+1) << "  ,  " << (int)src.at<unsigned char>(i*j + j+2)<<",  ";
		//		float cmax, cmin, cdelta;
		//		cmax = max({ r, g, b });
		//		cmin = min({ r, g, b });
		//		cdelta = cmax - cmin;
		//		v = cmax;
		//		if (cmax == 0)
		//			s = 0;
		//		else
		//			s = cdelta / cmax;
		//		if (cdelta == 0)
		//			h = 0;
		//		else if (cmax == r)
		//			h = 60.0 * (((g - b) / cdelta));
		//		else if (cmax == g)
		//			h = 60.0 * (((b - r) / cdelta) + 2.0);
		//		else if (cmax ==b)
		//			h = 60.0 * (((r - g) / cdelta) + 4.0);
		//		float x, y;
		//		x = s*cos(h*PI / 180.0);
		//		y = s*sin(h*PI / 180.0);
		//		if (h > 360)
		//			system("pause");
		//		//cout << h << "  ,  " << s << "  ,  " << v << "  ,      " <<x << "  ,  " <<y<< endl;
		//		hsv.push_back(Vec3f(h, s, v));
		//		hxy.push_back(Vec2f(x, y));

		//	}


		//**********************************HSV Value****************************//

		//**********************************Gaussian Map and cut****************//

		//imgtovertex(src);

		//addWeighted(sobelrstx, 0.8, kiredge, 0.2,0,sobelrstx);
		//addWeighted(sobelrsty, 0.8, kiredge, 0.2, 0, sobelrsty);
		//ldp*********************

		Mat tempseamy, tempseamx;
		Mat seamxy;
		//addWeighted(ldpimg, 10, ldpimg, 0, 0,ldpimg);
		//imshow("ldprank", ldpimg);
		//Mat thrx, thry;
		//threshold(sobelrstx,thrx,threshvalue,255,THRESH_OTSU);
		//threshold(sobelrsty, thry, threshvalue, 255, THRESH_OTSU);
		//imshow("thry", thry);
		//imshow("thrx", thrx);
		//Mat ldpx, ldpy;
		//addWeighted(ldpimg, 1, sobelrstx, 0.9, 0, ldpx);
		//addWeighted(ldpimg, 1, sobelrsty, 0.9, 0, ldpy);
		//imshow("ldpy", ldpy);
		//imshow("ldpx", ldpx);

		//test(sobelrstx, tempseamx, sobelrsty, tempseamy);		//******************************************************

		//ldp***********************

		//imshow("tempseamy", tempseamy);
		//imshow("tempseamx", tempseamx);
		/*
		// find path from 2 direction x y down left with 2 sobel rst x & y
		energy(sobelrsty,tempseamy);	//x direction with sobelrstx
		//imshow("tempseamy",tempseamy);

		transpose(sobelrstx, sobelrstx);	////y direction with sobelrsty
		flip(sobelrstx, sobelrstx, 3);
		energy(sobelrstx,tempseamx);
		transpose(tempseamx, tempseamx);
		flip(tempseamx, tempseamx, 0);
		//imshow("tempseamx", tempseamx);
		*/
		//seamxy = tempseamx + tempseamy;						//****************************************************
		//imshow("seamxy2sobel", seamxy);
		// find path from 2 direction x y down left with 2 sobel rst x & y


		// find path from 2 direction x y down left with merged sobel rst
		/*
		energy(sobelrst,tempseamy);	//x direction with sobelrst
		//imshow("tempseamy",tempseamy);

		energy(sobelrsttrans,tempseamx);		//y direction with sobelrst
		transpose(tempseamx, tempseamx);
		flip(tempseamx, tempseamx, 0);
		imshow("tempseamx", tempseamx);
		seamxy = tempseamx + tempseamy;
		imshow("seamxy1sobel", seamxy);
		// find path from 2 direction x y down left with merged sobel rst
		*/



		//**********************************Gaussian Map and cut****************//

		//**********************************CUTTINGtest*************************//
		//int pixel;
		//int x, y = 0;
		//for (int i = 0; i < src.rows; i++)
		//	for (int j = 0; j < src.cols; j++) {
		//		x = (i / 20)+1;
		//		y = (j / 20)+1;
		//		pixel = x*y + y;			
		//		label.push_back(pixel);
		//		//cout << label[i*src.cols + j] << endl;
		//	}
		//**********************************CUTTINGtest*************************//


		//**********************************dump label and draw seam on src pic*************************//
		/*dumplabel = Mat::zeros(src.rows, src.cols, CV_8UC3);

		for (int i = 0; i < src.rows; i++)
		{
		for (int j = 0; j < src.cols; j++)
		{
		dumplabel.at<Vec3b>(i*src.cols + j) = random_color[label[i*src.cols + j]];
		if (label[i*src.cols + j] != label[i*src.cols + j + 1]) {
		src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
		if (label[i*src.cols + j] != label[i*src.cols + j + src.cols]) {
		src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
		}
		}*/
		//imshow("dump", dumplabel);
		//imshow("rst", src);
		//**********************************dump label and draw seam on src pic*************************//

		//dumpsuperpixel(tempseamx, tempseamy, srcds , superpixels);
		//transpose(sobelrstx, sobelrstx);						//****************************************************
		//flip(sobelrstx, sobelrstx, 0);						//****************************************************
		vector<vector<int>> original;
		vector <int> initial;
		for (int x = 0; x < 1; x++) {
			original.push_back(initial);
			for (int y = 0; y < src.cols*src.rows; y++) {
				original.at(x).push_back(y);
			}
		}




		vector <meta> outmeta;
		vector <superpixel> supout;

		vector<vector<int>> rst1, rst2, rst3, rst4, rst5, rst6;

		//clock_t start0 = clock();
		firstseam(original, energy_x, energy_y, rst, outmeta, supout);
		//clock_t ends0 = clock();
		//cout << "Time elapsed 0 : " << (double)(ends0 - start0) / CLOCKS_PER_SEC << endl;
		//vec2txt(rst);
		//waitKey(0);

		/*
		clock_t start0 = clock();
		iteratorseam(original,sobelrstx,sobelrsty,rst);
		clock_t ends0 = clock();
		cout << "Time elapsed 0 : " << (double)(ends0 - start0) / CLOCKS_PER_SEC << endl;
		*/
		//clock_t start1 = clock();
		iteratorseam(rst, energy_x, energy_y, rst1, outmeta, supout);
		//itersp(rst,sobelrsty,sobelrstx,rst1,outmeta,supout);
		//clock_t ends1 = clock();
		//cout << "Time elapsed 1 : " << (double)(ends1 - start1) / CLOCKS_PER_SEC << endl;
		//vec2txt(rst1);
		//waitKey(0);
		//clock_t start2 = clock();
		iteratorseam(rst1, energy_x, energy_y, rst2, outmeta, supout);
		//clock_t ends2 = clock();
		//cout << "Time elapsed 2 : " << (double)(ends2 - start2) / CLOCKS_PER_SEC << endl;
		//vec2txt(rst2);
		//waitKey(0);
		//clock_t start3 = clock();
		iteratorseam(rst2, energy_x, energy_y, rst3, outmeta, supout);
		//clock_t ends3 = clock();
		//cout << "Time elapsed 3 : " << (double)(ends3 - start3) / CLOCKS_PER_SEC << endl;

		vec2txt(rst3);
		tempspcnt = tempspcnt + glogalspcounter;
		//waitKey(0);
		//clock_t start4 = clock();
		//iteratorseam(rst3, sobelrstx, sobelrsty, rst4);
		//clock_t ends4 = clock();
		//cout << "Time elapsed 4 : " << (double)(ends4 - start4) / CLOCKS_PER_SEC << endl;
		//clock_t start5 = clock();
		//iteratorseam(rst4, sobelrstx, sobelrsty, rst5);
		//clock_t ends5 = clock();
		//cout << "Time elapsed 5 : " << (double)(ends5 - start5) / CLOCKS_PER_SEC << endl;

		//vec2txt(superpixels);
		//waitKey(0);


		/*clock_t start44 = clock();
		vec2txt(rst);
		clock_t ends44 = clock();
		cout << "Time dump1 : " << (double)(ends44 - start44) / CLOCKS_PER_SEC << endl;
		waitKey(0);
		clock_t start55 = clock();
		vec2txt(rst1);
		clock_t ends55 = clock();
		cout << "Time dump2 : " << (double)(ends55 - start55) / CLOCKS_PER_SEC << endl;
		waitKey(0);
		clock_t start66 = clock();
		vec2txt(rst2);
		clock_t ends66 = clock();
		cout << "Time dump3 : " << (double)(ends66 - start66) / CLOCKS_PER_SEC << endl;
		waitKey(0);
		clock_t start77 = clock();
		vec2txt(rst3);
		clock_t ends77 = clock();
		cout << "Time dump4 : " << (double)(ends77 - start77) / CLOCKS_PER_SEC << endl;
		waitKey(0);*/
		//vec2txt(rst4);
		//waitKey(0);
		//vec2txt(rst5);
		//waitKey(0);

		cout << filename_image << "done" << endl;
		superpixels.shrink_to_fit();
		original.shrink_to_fit();
		rst.shrink_to_fit();
		rst1.shrink_to_fit();
		//rst2.shrink_to_fit();
		//rst3.shrink_to_fit();
		//rst4.shrink_to_fit();
		waitKey(1);
	}

	cout << "total sp NUM :" << (float)tempspcnt / 50 << endl;

	waitKey(0);
	//system("pause");
	return 0;
}

void colordownsample(const Mat& src, Mat& rst, int dsrate) {

	rst.create(src.size(), src.type());
	int nr = src.rows;
	int nc = src.cols;
	if (src.isContinuous() && rst.isContinuous()) {
		nr = 1;
		nc = nc*src.rows*src.channels();
	}
	for (int i = 0; i < nr; i++) {
		const uchar* inData = src.ptr<uchar>(i);
		uchar* Outdata = rst.ptr<uchar>(i);
		for (int j = 0; j < nc; j++) {
			*Outdata++ = *inData++ / dsrate*dsrate + dsrate / 2;
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

void energy(const Mat &src, Mat &rst) {

	Mat verenergy;
	verenergy.create(src.size(), CV_32FC1);
	Mat verdot;
	verdot.create(src.size(), CV_32FC1);

	int w = SP;
	double center = gaussian(0, w);
	int nr, nc;

	if (src.isContinuous() && verenergy.isContinuous()) {
		nr = 1;
		nc = src.cols*src.rows*src.channels();
	}


	for (int i = 0; i < nr; i++) {
		float* data = verenergy.ptr<float>(i);
		for (int j = 0; j < nc; j++) {
			int x = j%verenergy.cols;					// version 1  (from left)
			//int x = j%verenergy.cols +(SP - ((verenergy.cols%SP) / 2));		//version 2 (in the middle)
			int y = x%w;
			if (y == 0)
				*data++ = 0;
			else
				*data++ = ((gaussian(SP / 2 - y, w) / center));
		}
	}
	verenergy.convertTo(verenergy, CV_8U, 255); //convert to 0 ~ 255
	//verenergy.convertTo(verenergy, CV_32FC1);
	//src.convertTo(src, CV_32FC1);
	//imshow("verseam", verenergy);
	addWeighted(src, 2.7, verenergy, SPATIALWEIGHT, 0, verdot);


	//if (src.isContinuous() && verdot.isContinuous()) {
	//	nr = 1;
	//	nc = src.cols*src.rows*src.channels();
	//}

	//for (int i = 0; i < nr; i++) {
	//	float* data = verdot.ptr<float>(i);
	//	const uchar* inData = src.ptr<uchar>(i);
	//	const float* inweight = verenergy.ptr<float>(i);
	//	for (int j = 0; j < nc; j++) {
	//		*data++ = (float)(*inData++)*(*inweight++);
	//	}
	//}

	//imshow("verdot", verdot);

	Mat seammap, seamvalue;
	seamvalue = Mat::zeros(src.rows, src.cols, CV_32S);;
	seammap = Mat::zeros(src.rows, src.cols, CV_8U);

	//vector<int> temprow;							//may cahange to callalo    performence up?
	const uchar* inData = verdot.ptr<uchar>(0);
	uchar* tempdata = seammap.ptr<uchar>(0);
	int* tempvalue = seamvalue.ptr<int>(0);


	for (int i = 0; i < 1 + src.cols / w; i++) {		//colon block
		for (int j = 0; j < src.rows; j++) {		//rows

			//*******************************************************************************************************************************************

			//Method 1 find MAX each col(row) 
			//******************1***************
			//***************1******************
			/*
			int maxindex = 0;
			int maxtemp = 0;


			for (int x = 0; x < w; x++) {			//label conter
			//temprow.push_back((int)inData[j*src.cols + i*w + x]);
			//cout << (int)inData[j*src.cols + i*w + x] << endl;
			if ((int)inData[j*src.cols + i*w + x] >= maxtemp) {
			maxindex = j*src.cols + i*w + x;
			maxtemp = (int)inData[j*src.cols + i*w + x];
			//tempdata[j*src.cols + i*w + x] =255;
			}
			}


			tempdata[maxindex] = 255;
			temprow.clear();

			*/
			//******************1***************
			//***************1******************
			//Method 1 find MAX each col(row)

			//*******************************************************************************************************************************************


			//Method 2 find MAX and the next col find MAXINDEX adherence
			//******************1***************
			//*****************000**************
			/*
			int maxindex;
			int maxtemp=0;

			if (j == 0) {
			maxindex = 0;

			for (int x = 0; x < w; x++) {			//label conter
			//temprow.push_back((int)inData[j*src.cols + i*w + x]);
			//cout << (int)inData[j*src.cols + i*w + x] << endl;
			if ((int)inData[j*src.cols + i*w + x] >= maxtemp) {
			maxindex = j*src.cols + i*w + x;
			maxtemp = (int)inData[j*src.cols + i*w + x];
			//tempdata[j*src.cols + i*w + x] =255;
			}
			}
			}
			else {

			int maxindex2=0;
			for (int x = 0; x < 3; x++) {
			if ((int)inData[maxindex - 1 + x + src.cols] >= maxtemp) {
			maxindex2 = maxindex - 1 + x + src.cols;
			maxtemp = (int)inData[maxindex2];
			}
			}
			maxindex = maxindex2;
			}



			tempdata[maxindex] = 255;
			temprow.clear();

			*/
			//******************1***************
			//*****************000**************
			//Method 2 find MAX and the next col find MAXINDEX adherence 

			//*******************************************************************************************************************************************

			//Method 3 max accumulate
			/*
			int maxindex;
			int maxtemp = 0;
			int index2;


			if (j == 0) {
			maxindex = 0;
			//index2 = 0;
			for (int x = 0; x < w; x++) {			//label conter

			tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x];
			if ((int)inData[j*src.cols + i*w + x] >= maxtemp) {
			maxindex = j*src.cols + i*w + x;
			//index2 = maxindex;
			maxtemp = (int)inData[j*src.cols + i*w + x];
			}
			}
			}
			else {
			maxindex = 0;
			for (int x = 0; x < w; x++) {

			int vmax = 0;
			if (x == 0) {
			vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });
			//cout << "0" << endl;
			//cout << (int)tempvalue[(j - 1)*src.cols + w*i + x] << "," << (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] << endl;
			//cout << vmax << endl;
			}
			else if (x == (w - 1)) {
			vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x] });
			//cout << "w" << endl;
			//cout << (int)tempvalue[(j - 1)*src.cols + w*i + x - 1] << "," << (int)tempvalue[(j - 1)*src.cols + w*i + x] << endl;
			//cout << vmax << endl;
			}
			else {
			vmax = max({(int)tempvalue[(j - 1)*src.cols + w*i + x -1], (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1]});
			//cout << "3" << endl;
			//cout << (int)tempvalue[(j - 1)*src.cols + w*i + x -1 ] <<","<< (int)tempvalue[(j - 1)*src.cols + w*i + x] << "," << (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] << endl;
			//cout << vmax << endl;
			}
			tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x]+vmax;

			if ((int)tempvalue[j*src.cols + i*w + x] >= maxtemp) {
			maxindex = j*src.cols + i*w + x;
			maxtemp = (int)tempvalue[j*src.cols + i*w + x];
			}

			}


			}
			if (j == 0) {

			maxindex = maxindex;
			index2 = maxindex;
			}
			else {
			int maxindex2 = 0;
			maxtemp = 0;
			for (int x = 0; x < 3; x++) {
			if ((int)tempvalue[index2 - 1 + x +src.cols] >= maxtemp) {
			maxindex2 = index2 - 1 + x + src.cols;
			maxtemp = (int)tempvalue[maxindex2];
			}
			}

			maxindex = maxindex2;
			index2 = maxindex;

			}

			tempdata[maxindex] = 255;
			//temprow.clear();
			*/
			//Method 3 max accumulate

			//*******************************************************************************************************************************************

			//Method 4 original

			int maxindex;
			int maxtemp = 0;
			int index2;


			if (j == 0) {
				maxindex = 0;

				if (i < src.cols / w + 1) {

					if (i == 0) {

						if ((src.cols%w) / 2 == 0) {
							for (int x = 0; x < w; x++) {			//label conter
								tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x];

								if ((int)inData[j*src.cols + i*w + x] >= maxtemp) {
									maxindex = j*src.cols + i*w + x;

									maxtemp = (int)inData[j*src.cols + i*w + x];
								}
							}

						}
						else
							for (int x = 0; x < (src.cols%w) / 2; x++) {			//label conter
								tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x];
								//cout << "test" << j*src.cols + i*w + x << endl;
								if ((int)inData[j*src.cols + i*w + x] >= maxtemp) {
									maxindex = j*src.cols + i*w + x;

									maxtemp = (int)inData[j*src.cols + i*w + x];
								}
							}



					}
					else{
						for (int x = 0; x < w; x++) {			//label conter

							tempvalue[j*src.cols + (i - 1)*w + x + (src.cols%w) / 2] = inData[j*src.cols + (i - 1)*w + x + (src.cols%w) / 2];
							if ((int)inData[j*src.cols + i*w + x] >= maxtemp) {
								maxindex = j*src.cols + i*w + x;

								maxtemp = (int)inData[j*src.cols + i*w + x];
							}
						}
					}


				}

				else {
					for (int x = 0; x < (src.cols%w) / 2; x++) {			//label conter
						tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x];
						if ((int)inData[j*src.cols + i*w + x] >= maxtemp) {
							maxindex = j*src.cols + i*w + x;

							maxtemp = (int)inData[j*src.cols + i*w + x];
						}
					}

				}

			}

			else {
				maxindex = 0;

				if (i < src.cols / w) {

					if (i == 0) {
						if ((src.cols%w) / 2 == 0) {
							for (int x = 0; x < w; x++) {

								int vmax = 0;
								if (x == 0) {
									vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });

								}
								else if (x == (w - 1)) {
									vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x] });

								}
								else {
									vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });

								}
								tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x] + vmax;

								if ((int)tempvalue[j*src.cols + i*w + x] >= maxtemp) {
									maxindex = j*src.cols + i*w + x;
									maxtemp = (int)tempvalue[j*src.cols + i*w + x];
								}

							}
						}
						else {
							for (int x = 0; x < (src.cols%w) / 2; x++) {

								int vmax = 0;
								if (x == 0) {
									vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });

								}
								else if (x == (w - 1)) {
									vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x] });

								}
								else {
									vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });

								}
								tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x] + vmax;

								if ((int)tempvalue[j*src.cols + i*w + x] >= maxtemp) {
									maxindex = j*src.cols + i*w + x;
									maxtemp = (int)tempvalue[j*src.cols + i*w + x];
								}

							}
						}


					}

					else {
						for (int x = 0; x < w; x++) {

							int vmax = 0;
							if (x == 0) {
								vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });

							}
							else if (x == (w - 1)) {
								vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x] });
							}
							else {
								vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });
							}
							tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x] + vmax;

							if ((int)tempvalue[j*src.cols + i*w + x] >= maxtemp) {
								maxindex = j*src.cols + i*w + x;
								maxtemp = (int)tempvalue[j*src.cols + i*w + x];
							}
						}
					}
				}
				else {
					for (int x = 0; x < (src.cols%w) / 2; x++) {

						int vmax = 0;
						if (x == 0) {
							vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });

						}
						else if (x == (w - 1)) {
							vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x] });

						}
						else {
							vmax = max({ (int)tempvalue[(j - 1)*src.cols + w*i + x - 1], (int)tempvalue[(j - 1)*src.cols + w*i + x], (int)tempvalue[(j - 1)*src.cols + w*i + x + 1] });

						}
						tempvalue[j*src.cols + i*w + x] = inData[j*src.cols + i*w + x] + vmax;

						if ((int)tempvalue[j*src.cols + i*w + x] >= maxtemp) {
							maxindex = j*src.cols + i*w + x;
							maxtemp = (int)tempvalue[j*src.cols + i*w + x];
						}
					}
				}
			}

			if (j == 0) {

				maxindex = maxindex;
				index2 = maxindex;
			}
			else {
				int maxindex2 = 0;
				maxtemp = 0;
				for (int x = 0; x < 3; x++) {
					if ((int)tempvalue[index2 - 1 + x + src.cols] >= maxtemp) {
						maxindex2 = index2 - 1 + x + src.cols;
						maxtemp = (int)tempvalue[maxindex2];
					}
				}

				maxindex = maxindex2;
				index2 = maxindex;

			}

			//tempdata[maxindex] = 255;

			//Method 4 original

			//Method 4 mod



			//Method 4 mod


		}

	}
	//imshow("seammp", seammap);

	//Method 4 continue traceback
	vector<int> rowvalue;
	int index;
	int indexposition;
	int returnindex;

	//****** original path trace back*********************
	//for (int i = 0; i < src.cols / w; i++) {
	//	for (int j = src.rows; j >0; j--) {
	//		//********last rows
	//		if (j == src.rows){
	//			for (int x = 0; x < w; x++) {
	//				index = (j - 1)*src.cols + i*w + x;
	//				rowvalue.push_back(seamvalue.at<int>(index));
	//			}
	//			int tempmax;
	//			auto biggest = max_element(rowvalue.begin(), rowvalue.end());
	//			tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
	//			indexposition = distance(begin(rowvalue), biggest);
	//			returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector
	//			//cout << "indexposition" << indexposition << endl;
	//		}
	//		else {
	//			//left boundry
	//			if (indexposition <= 0) {
	//			//if (0) {
	//				for (int x = indexposition; x < indexposition+2; x++) {
	//					index = (j - 1)*src.cols + i*w + x;
	//					rowvalue.push_back(seamvalue.at<int>(index));
	//				}
	//				int tempmax;
	//				auto biggest = max_element(rowvalue.begin(), rowvalue.end());
	//				tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
	//				indexposition = indexposition +distance(begin(rowvalue), biggest);
	//				returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector
	//				
	//			}
	//			//right boundry
	//			else if (indexposition >= w - 1) {
	//			//else if (0) {
	//				for (int x = indexposition-1; x < indexposition +1; x++) {
	//					index = (j - 1)*src.cols + i*w + x;
	//					rowvalue.push_back(seamvalue.at<int>(index));
	//				}
	//				int tempmax;
	//				auto biggest = max_element(rowvalue.begin(), rowvalue.end());
	//				tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
	//				indexposition = indexposition - 1+distance(begin(rowvalue), biggest);
	//				returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector
	//				
	//			}
	//			//midle
	//			else {
	//				for (int x = indexposition - 1; x < indexposition + 2; x++) {
	//					index = (j - 1)*src.cols + i*w + x;
	//					rowvalue.push_back(seamvalue.at<int>(index));
	//				}
	//				int tempmax;
	//				auto biggest = max_element(rowvalue.begin(), rowvalue.end());
	//				tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
	//				indexposition = indexposition-1+distance(begin(rowvalue), biggest);
	//				returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector
	//			}
	//		}

	//		//for (int x = 0; x < w; x++) {
	//		//	index = (j - 1)*src.cols + i*w + x;
	//		//	rowvalue.push_back(seamvalue.at<int>(index));
	//		//}
	//		//int tempmax;
	//		//auto biggest = max_element(rowvalue.begin(), rowvalue.end());
	//		//tempmax = *max_element( rowvalue.begin(), rowvalue.end() );  //find the max element in vector
	//		//
	//		////find max element index  *************************************************************
	//		////method 1
	//		//returnindex = (j - 1)*src.cols + i*w + distance(begin(rowvalue), biggest);

	//		//method 2
	//		/*
	//		for (int x = 0; x < w; x++) {
	//			index = (j - 1)*src.cols + i*w + x;
	//			if (tempmax == seamvalue.at<int>(index))
	//				returnindex = index;
	//			else
	//				returnindex = returnindex;
	//		}
	//		*/
	//		//find max element index  *************************************************************
	//		rowvalue.clear();			
	//		tempdata[returnindex] = 255;
	//	}
	//}
	//****** original path trace back*********************

	//****** trace back with krisch weight

	for (int i = 0; i < src.cols / w; i++) {
		for (int j = src.rows; j >0; j--) {
			//********last rows
			if (j == src.rows) {
				for (int x = 0; x < w; x++) {
					index = (j - 1)*src.cols + i*w + x;
					rowvalue.push_back(seamvalue.at<int>(index));
				}
				int tempmax;
				auto biggest = max_element(rowvalue.begin(), rowvalue.end());
				tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
				indexposition = distance(begin(rowvalue), biggest);
				returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector
				//cout << "indexposition" << indexposition << endl;
			}
			else {
				//left boundry
				if (indexposition <= 0) {
					//if (0) {
					for (int x = indexposition; x < indexposition + 2; x++) {
						index = (j - 1)*src.cols + i*w + x;
						//if (weight[j-1][i*w+x][3-(x-indexposition)]>= seamvalue.at<int>(index)) {
						//	rowvalue.push_back(weight[j - 1][i*w + x][3 - (x - indexposition)] +seamvalue.at<int>(index));
						//}
						//else {
						rowvalue.push_back(seamvalue.at<int>(index));
						//}


					}
					int tempmax;
					auto biggest = max_element(rowvalue.begin(), rowvalue.end());
					tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
					indexposition = indexposition + distance(begin(rowvalue), biggest);
					returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector

				}
				//right boundry
				else if (indexposition >= w - 1) {
					//else if (0) {
					for (int x = indexposition - 1; x < indexposition + 1; x++) {
						index = (j - 1)*src.cols + i*w + x;
						//if (weight[j - 1][i*w + x][3 - (x - indexposition)] >= seamvalue.at<int>(index)) {
						//	rowvalue.push_back(weight[j - 1][i*w + x][3 - (x - indexposition)] + seamvalue.at<int>(index));
						//}
						//else {
						rowvalue.push_back(seamvalue.at<int>(index));
						//}
					}
					int tempmax;
					auto biggest = max_element(rowvalue.begin(), rowvalue.end());
					tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
					indexposition = indexposition - 1 + distance(begin(rowvalue), biggest);
					returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector

				}
				//midle
				else {
					for (int x = indexposition - 1; x < indexposition + 2; x++) {
						index = (j - 1)*src.cols + i*w + x;
						//if (weight[j-1][i*w+x][3-(x-indexposition)]>= seamvalue.at<int>(index)) {
						//	rowvalue.push_back(weight[j - 1][i*w + x][3 - (x - indexposition)] +seamvalue.at<int>(index));
						//}
						//else {
						rowvalue.push_back(seamvalue.at<int>(index));
						//}
					}
					int tempmax;
					auto biggest = max_element(rowvalue.begin(), rowvalue.end());
					tempmax = *max_element(rowvalue.begin(), rowvalue.end());  //find the max element in vector
					indexposition = indexposition - 1 + distance(begin(rowvalue), biggest);
					returnindex = (j - 1)*src.cols + i*w + indexposition; //find the max position in vector
				}
			}
			rowvalue.clear();
			tempdata[returnindex] = 255;
		}
	}



	//****** trace back with krisch weight



	//imshow("seammp", seammap);
	rst.create(seammap.size(), seammap.type());
	rst = seammap.clone();

	//Method 4 continue traceback


}
int labvarweight(int i, int j) {
	int l, a, b, v;
	l = src_labptr[(i - 1) * 3 + 0] - src_labptr[(j - 1) * 3 + 0];
	//cout <<"l:" <<l << endl;
	a = src_labptr[(i - 1) * 3 + 1] - src_labptr[(j - 1) * 3 + 1];
	//cout << "a:" << a << endl;
	b = src_labptr[(i - 1) * 3 + 2] - src_labptr[(j - 1) * 3 + 2];
	//cout << "b:" << b << endl;
	l = abs(l);
	a = abs(a);
	b = abs(b);
	v = ((l + a + b) / 3) / 10;
	//cout << "l + a + b:" << l + a + b << endl;
	v = 25 - v;
	return v;
}


void imgtovertex(Mat src) {
	char filename[] = "pixel.txt";
	fstream fp;
	fp.open(filename, ios::out);
	if (!fp) {
		cout << "Fail to open file: " << filename << endl;
	}
	//	cout << "File Descriptor: " << fp << endl;
	fp << "vertices " << (src.cols*src.rows) << endl;//寫入字串
	fp << "start 1 " << endl;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (i == 0 && j == 0) {
				fp << " edge source " << j + 1 << " target " << j + 2 << " weight " << labvarweight(j + 1, j + 2) << endl;
				fp << " edge source " << j + 1 << " target " << j + 1 + src.cols << " weight " << labvarweight(j + 1, j + 1 + src.cols) << endl;
				fp << " edge source " << j + 1 << " target " << j + 1 + src.cols + 1 << " weight " << labvarweight(j + 1, j + 1 + src.cols + 1) << endl;
			}
			else if (i == (src.rows - 1) && j == 0)
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + 1 + (i*src.cols) + 1 << " weight " << labvarweight(j + 1 + (i*src.cols), j + 1 + (i*src.cols) + 1) << endl;
			else if (i == 0 && j == (src.cols - 1)) {
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + 1 + ((i + 1)*src.cols) << " weight " << labvarweight(j + 1 + (i*src.cols), j + 1 + ((i + 1)*src.cols)) << endl;
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols)) << endl;
			}

			else if (j == 0 && i != 0) {
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + 1 + (i*src.cols) + 1 << " weight " << labvarweight(j + 1 + (i*src.cols), j + 1 + (i*src.cols) + 1) << endl;
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols + 1) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols + 1)) << endl;
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols + 2) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols + 2)) << endl;
			}
			else if (j == (src.cols - 1) && i != 0) {
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols)) << endl;
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols + 1) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols + 1)) << endl;
			}
			else if (i == (src.rows - 1) && j != (src.cols - 1))
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + 1 + (i*src.cols) + 1 << " weight " << labvarweight(j + 1 + (i*src.cols), j + 1 + (i*src.cols) + 1) << endl;
			else if (i == (src.rows - 1) && j == (src.cols - 1))
				fp << "fuck" << endl;
			else {
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + 1 + (i*src.cols) + 1 << " weight " << labvarweight(j + 1 + (i*src.cols), j + 1 + (i*src.cols) + 1) << endl;
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols)) << endl;
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols + 1) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols + 1)) << endl;
				fp << " edge source " << j + 1 + (i*src.cols) << " target " << j + ((i + 1)*src.cols + 2) << " weight " << labvarweight(j + 1 + (i*src.cols), j + ((i + 1)*src.cols + 2)) << endl;
			}



		}
	}
	fp << " end " << endl;
	fp.close();

}

void dumpsuperpixel(const Mat& seammapx, const Mat& seammapy, const Mat& src, vector <vector <int>> & superpixels) {
	//vector <int> labelx, labely;
	vector <superpixel> super;
	vector <int> supercnt;
	unsigned char *seamx = (unsigned char*)(seammapx.data);
	unsigned char *seamy = (unsigned char*)(seammapy.data);

	//labelx.reserve(seammapx.cols*seammapx.rows);
	//labely.reserve(seammapy.cols*seammapy.rows);

	Mat labelxx, labelyy;
	labelxx = Mat::zeros(src.rows, src.cols, CV_8U);
	labelyy = Mat::zeros(src.rows, src.cols, CV_8U);
	unsigned char *labelx = (unsigned char*)(labelxx.data);
	unsigned char *labely = (unsigned char*)(labelyy.data);


	bool seamflag;
	int cnt, xcnt, ycnt;
	int i, j;


	/*
	//char filename[] = "result/labels.txt";
	fstream fp;
	fp.open(filename_label, ios::out);
	if (!fp) {
	cout << "Fail to open file: " << filename_label << endl;
	}
	*/
	//***labeling y axis****

	for (i = 0; i < seammapy.rows; i++) {
		cnt = 1;
		seamflag = 0;
		for (j = 0; j < seammapy.cols; j++) {
			if (seamflag == 1) {
				if (seamy[i*seammapy.cols + j] == 0) {
					labely[i*seammapy.cols + j] = cnt;
					seamflag = 0;
				}
				if (seamy[i*seammapy.cols + j] == 255) {
					cnt++;
					labely[i*seammapy.cols + j] = cnt;
					seamflag = 0;
				}

			}
			else {
				if (seamy[i*seammapy.cols + j] == 0) {
					labely[i*seammapy.cols + j] = cnt;
					seamflag = 0;
				}

				if (seamy[i*seammapy.cols + j] == 255) {
					cnt++;
					labely[i*seammapy.cols + j] = cnt;
					seamflag = 1;
				}
			}
		}
		ycnt = cnt;
	}

	//***labeling x axis****
	for (i = 0; i < seammapx.cols; i++) {
		cnt = 1;
		seamflag = 0;
		for (j = 0; j < seammapx.rows; j++) {
			if (seamflag == 1) {
				if (seamx[j*seammapx.cols + i] == 0) {
					labelx[j*seammapx.cols + i] = cnt;
					seamflag = 0;
				}
				if (seamx[j*seammapx.cols + i] == 255) {
					cnt++;
					labelx[j*seammapx.cols + i] = cnt;
					seamflag = 0;
				}

			}
			else {
				if (seamx[j*seammapx.cols + i] == 0) {
					labelx[j*seammapx.cols + i] = cnt;
					seamflag = 0;
				}

				if (seamx[j*seammapx.cols + i] == 255) {
					cnt++;
					labelx[j*seammapx.cols + i] = cnt;
					seamflag = 1;
				}
			}
		}
		xcnt = cnt;
	}
	//imshow("labelx", labelxx);
	//imshow("labely", labelyy);
	int totalsuperpixels = xcnt*ycnt;
	for (int i = 0; i < totalsuperpixels; i++) {
		superpixels.push_back(supercnt);
	}

	vector<Vec3b>random_color;
	for (int rr = 0; rr < 500000; rr++)
		random_color.push_back(Vec3b(rand() % 255, rand() % 255, rand() % 255));

	Mat superpix, superpixx, superpixxx;
	superpix = Mat::zeros(src.rows, src.cols, CV_32F);
	float *superptr = (float*)(superpix.data);
	superpixx = Mat::zeros(src.rows, src.cols, CV_8UC3);
	unsigned char *superptrr = (unsigned char*)(superpixx.data);
	superpixxx = Mat::zeros(src.rows, src.cols, CV_8UC3);
	unsigned char *superptrrr = (unsigned char*)(superpixxx.data);
	vector<avg_color_> avgcolor;
	avgcolor.reserve(50000);
	avgcolor.resize(50000);  //initial to zero value

	//random color label
	//superpix contains labels 
	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			//super.push_back(superpixel());
			//super.back().label = labelx[i*src.cols + j]*labely[i*src.cols+j];
			if (labelx[i*src.cols + j] == 0 || labely[i*src.cols + j] == 0) {
				superptr[i*src.cols + j] = 0;
				/*if (j==0)
				fp << (int)superpix.at<float>(i-1, j) -1<< " ";
				else {
				if (((int)superpix.at<float>(i, j - 1)) != 0)
				fp << (int)superpix.at<float>(i, j - 1) - 1 << "a ";
				else if (((int)superpix.at<float>(i - 1, j)) != 0)
				fp << (int)superpix.at<float>(i - 1, j) - 1 << "b ";
				else if (((int)superpix.at<float>(i - 1, j - 1)) != 0)
				fp << (int)superpix.at<float>(i - 1, j - 1) - 1 << "c ";
				else if (((int)superpix.at<float>(i - 1, j + 1)) != 0)
				fp << (int)superpix.at<float>(i - 1, j + 1) - 1 << "d ";
				else if (((int)superpix.at<float>(i , j + 1)) != 0)
				fp << (int)superpix.at<float>(i , j + 1) - 1 << "e ";
				else if (((int)superpix.at<float>(i + 1, j + 1)) != 0)
				fp << (int)superpix.at<float>(i + 1, j + 1) - 1 << "f ";
				else if (((int)superpix.at<float>(i + 1, j )) != 0)
				fp << (int)superpix.at<float>(i - 1, j ) - 1 << "g ";
				else
				fp << (int)superpix.at<float>(i + 1, j - 1) - 1 << "h ";


				}
				*/
			}
			else {
				superptr[i*src.cols + j] = ycnt*(labelx[i*src.cols + j] - 1) + labely[i*src.cols + j];
				superpixels.at(ycnt*(labelx[i*src.cols + j] - 1) + labely[i*src.cols + j] - 1).push_back(i*src.cols + j);
				superpixx.at<Vec3b>(i*src.cols + j) = random_color[superptr[i*src.cols + j]];
				avgcolor[superptr[i*src.cols + j]].cnt = avgcolor[superptr[i*src.cols + j]].cnt + 1;
				avgcolor[superptr[i*src.cols + j]].r = avgcolor[superptr[i*src.cols + j]].r + (int)src.at<Vec3b>(i, j)[2];
				avgcolor[superptr[i*src.cols + j]].g = avgcolor[superptr[i*src.cols + j]].g + (int)src.at<Vec3b>(i, j)[1];
				avgcolor[superptr[i*src.cols + j]].b = avgcolor[superptr[i*src.cols + j]].b + (int)src.at<Vec3b>(i, j)[0];
				//fp << (int)superpix.at<float>(i, j)-1 << " ";
			}
		}
		//fp << endl;
	}
	/*
	for (i = 0; i < src.rows; i++) {
	for (j = 0; j < src.cols; j++) {
	if (labelx[i*src.cols + j] == 0 || labely[i*src.cols + j] == 0) {
	if (j == 0) {
	if ((int)superpix.at<float>(i , j+1) != 0)
	fp << (int)superpix.at<float>(i , j+1) - 1 << " ";		//right
	else if ((int)superpix.at<float>(i + 1, j+1) != 0)
	fp << (int)superpix.at<float>(i + 1, j+1) - 1 << " ";	//right down
	else if ((int)superpix.at<float>(i+1, j - 1) != 0)
	fp << (int)superpix.at<float>(i+1, j -1) - 1 << " ";	//right up
	else if ((int)superpix.at<float>(i + 1, j - 1) != 0)
	fp << (int)superpix.at<float>(i + 1, j ) - 1 << " ";	// down
	else if ((int)superpix.at<float>(i -1, j ) != 0)
	fp << (int)superpix.at<float>(i - 1, j ) - 1 << " ";	//up
	else if ((int)superpix.at<float>(i  , j+2 ) != 0)
	fp << (int)superpix.at<float>(i , j+2 ) - 1 << " ";		//rr
	else if ((int)superpix.at<float>(i-1, j + 2) != 0)
	fp << (int)superpix.at<float>(i - 1, j+2 ) - 1 << " ";	//rru
	else
	fp << (int)superpix.at<float>(i + 1, j + 2) - 1 << " ";	//rrd


	}

	else {
	if (((int)superpix.at<float>(i, j - 1)) != 0)
	fp << (int)superpix.at<float>(i, j - 1) - 1 << " ";		//left
	else if (((int)superpix.at<float>(i, j + 1)) != 0)
	fp << (int)superpix.at<float>(i, j + 1) - 1 << " ";		//right
	else if (((int)superpix.at<float>(i - 1, j)) != 0)
	fp << (int)superpix.at<float>(i - 1, j) - 1 << " ";		//up
	else if (((int)superpix.at<float>(i - 1, j - 1)) != 0)
	fp << (int)superpix.at<float>(i - 1, j - 1) - 1 << " ";	//left up
	else if (((int)superpix.at<float>(i - 1, j + 1)) != 0)
	fp << (int)superpix.at<float>(i - 1, j + 1) - 1 << " ";	//right up
	else if (((int)superpix.at<float>(i + 1, j + 1)) != 0)
	fp << (int)superpix.at<float>(i + 1, j + 1) - 1 << " ";	//right down
	else if (((int)superpix.at<float>(i + 1, j)) != 0)
	fp << (int)superpix.at<float>(i + 1, j) - 1 << " ";		//down
	else if (((int)superpix.at<float>(i + 1, j-1)) != 0)
	fp << (int)superpix.at<float>(i + 1, j - 1) - 1 << " ";	//left down
	else if (((int)superpix.at<float>(i -2 , j )) != 0)
	fp << (int)superpix.at<float>(i -2, j ) - 1 << " ";		//upup
	else if (((int)superpix.at<float>(i - 2, j+1)) != 0)
	fp << (int)superpix.at<float>(i - 2, j+1) - 1 << " ";	//upup
	else if (((int)superpix.at<float>(i - 2, j-1)) != 0)
	fp << (int)superpix.at<float>(i - 2, j-1) - 1 << " ";	//upup
	else if (((int)superpix.at<float>(i + 2, j)) != 0)
	fp << (int)superpix.at<float>(i + 2, j) - 1 << " ";		//upup
	else
	fp << (int)superpix.at<float>(i + 2, j+1) - 1 << " ";	//upup

	}
	}
	else {
	fp << (int)superpix.at<float>(i, j) - 1 << " ";
	}


	}
	fp << endl;
	}


	fp.close();

	*/
	for (i = 0; i < avgcolor.size(); i++) {
		if (avgcolor[i].cnt != 0) {
			avgcolor[i].r = avgcolor[i].r / avgcolor[i].cnt;
			avgcolor[i].g = avgcolor[i].g / avgcolor[i].cnt;
			avgcolor[i].b = avgcolor[i].b / avgcolor[i].cnt;
		}
	}
	//avgcolr dump
	for (i = 0; i < src.rows; i++)
		for (j = 0; j < src.cols; j++) {
			Vec3b pix;
			pix[2] = avgcolor[superptr[i*src.cols + j]].r;
			pix[1] = avgcolor[superptr[i*src.cols + j]].g;
			pix[0] = avgcolor[superptr[i*src.cols + j]].b;


			superpixxx.at<Vec3b>(i, j) = pix;

		}



	//imshow("superpixel", superpix);
	//imshow("dump", superpixx);
	imshow("colordump", superpixxx);
	avgcolor.clear();
	random_color.clear();
	//labelx.clear();
	//labely.clear();
}
void test(Mat& x, Mat& seamx, Mat& y, Mat& seamy) {
	energy(y, seamy);	//x direction with sobelrstx
	//imshow("tempseamy",tempseamy); 

	transpose(x, x);	////y direction with sobelrsty
	flip(x, sobelrstx, 3);
	energy(x, seamx);
	transpose(seamx, seamx);
	flip(seamx, seamx, 0);


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

// method 1 cause slope cut
/*
void superpixelsplit(vector<vector<int>> sp,vector <Rect> &roirect, Mat & energy_y, Mat & energy_x,vector <vector<int>> &rst) {
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
roi.convertTo(roi,CV_32F);
ptrmask = mask.data;
ptrroi = roi.ptr<float>();
}

Dynamic_map_y_value.resize(roirect.at(x).height*roirect.at(x).width);
Dynamic_seam_y.resize(roirect.at(x).height);
for (int i = 0; i < roirect.at(x).height; i++) {
for (int j = 0; j < roirect.at(x).width; j++) {
ptrroi[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] * weightmap_y[j];
}
}

for (int i = 0; i < roirect.at(x).height; i++) {
for (int j = 0; j < roirect.at(x).width; j++) {

//First row value from weight roi
if (i == 0) {
if (ptrmask[j] == 255)	//check the pixel is flagged
{
Dynamic_map_y_value[j] = ptrroi[j];
//cout << "Dynamic_map_y_value[j]" << Dynamic_map_y_value[j] << endl;
}
else
continue;
}
//Following rows find the largest from above connected to itself
else {
if (ptrmask[i*roirect.at(x).width + j] == 255) {
if (j == 0) {	//left boundary
if (Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j] > Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1])
Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j];
else
Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1];
}
if (j == roirect.at(x).width) { //right  boundary
if (Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j-1] > Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j ])
Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j-1];
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
Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j ];
else
Dynamic_map_y_value[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] + Dynamic_map_y_value[(i - 1)*roirect.at(x).width + j + 1];
}

}
else
continue;

}
}
}
weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
//*****************show the weight map*********************
for (int y = 0; y < Dynamic_map_y_value.size(); y++) {

//weimap.create(mask.size(), CV_32F);
//weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
float * ptrweimap = weimap.ptr<float>(0);
ptrweimap[y] = Dynamic_map_y_value[y];
//imshow("weimap", weimap);
}
//*****************show the weight map*********************

if ((roirect.at(x).width < 10) || (roirect.at(x).width == 0)) {	//condition to find seams
Dynamic_seam_sp.at(x)=(Dynamic_seam_y);
//cout << "pushback null y DSY " << roirect.at(x).width << endl;
}
else {
//from last row
int j;
for (j = 0; j < roirect.at(x).width ; j++) {				//*************************-1???************************************
if (Dynamic_map_y_value[(roirect.at(x).height - 1)*roirect.at(x).width + Dynamic_seam_y.at(roirect.at(x).height - 1)] < Dynamic_map_y_value[(roirect.at(x).height - 1)*roirect.at(x).width + j]) {
Dynamic_seam_y.at(roirect.at(x).height - 1) = j;
//cout << "Dynamic_seam_y.at(roirect.at(x).height - 1)" << Dynamic_seam_y.at(roirect.at(x).height - 1) << endl;
}
}
for (int i = roirect.at(x).height - 2; i >= 0; i--) {
//cout << "i:" << i << endl;
j = Dynamic_seam_y.at(i+1);
//cout << "j:" << j << endl;
//left boundary
if (j == 0) {
if (Dynamic_map_y_value[i*roirect.at(x).width + j] <= Dynamic_map_y_value[i*roirect.at(x).width + j + 1])
Dynamic_seam_y.at(i) = j+1;
else
Dynamic_seam_y.at(i) = j ;
}
//right boundary
else if (j == roirect.at(x).width - 1) {
if (Dynamic_map_y_value[i*roirect.at(x).width + j] <= Dynamic_map_y_value[i*roirect.at(x).width + j - 1])
Dynamic_seam_y.at(i) = j-1;
else
Dynamic_seam_y.at(i) = j;
}
//middle
//if((j!= roirect.at(x).width - 1)&&(j!=0)) {
else{
if (Dynamic_map_y_value[i*roirect.at(x).width + j] < Dynamic_map_y_value[i*roirect.at(x).width + j + 1]) {
if (Dynamic_map_y_value[i*roirect.at(x).width + j-1] > Dynamic_map_y_value[i*roirect.at(x).width + j + 1])
Dynamic_seam_y.at(i) = j-1;
else
Dynamic_seam_y.at(i) = j + 1;
}
else {
if (Dynamic_map_y_value[i*roirect.at(x).width + j - 1] > Dynamic_map_y_value[i*roirect.at(x).width + j])
Dynamic_seam_y.at(i) = j - 1;
else
Dynamic_seam_y.at(i) = j ;
}
}

}
//cout << "x :" << x << endl;
Dynamic_seam_sp.at(x)=(Dynamic_seam_y);
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

Dynamic_map_x_value.resize(roirect.at(x).height*roirect.at(x).width);
Dynamic_seam_x.resize(roirect.at(x).width);
for (int i = 0; i < roirect.at(x).height; i++) {
for (int j = 0; j < roirect.at(x).width; j++) {
ptrroi[i*roirect.at(x).width + j] = ptrroi[i*roirect.at(x).width + j] * weightmap_x[i];
}
}

for (int i = 0; i < roirect.at(x).width; i++) {
for (int j = 0; j < roirect.at(x).height; j++) {

//First col value from weight roi
if (i == 0) {
if (ptrmask[j*roirect.at(x).width] == 255)	//check the pixel is flagged
{
Dynamic_map_x_value[j*roirect.at(x).width] = ptrroi[j*roirect.at(x).width];
//cout << "Dynamic_map_x_value[j*roirect.at(x).width]" << Dynamic_map_x_value[j*roirect.at(x).width] << endl;
}
else
continue;
}
//Following cols find the largest from above connected to itself
else {
if (ptrmask[j*roirect.at(x).width + i] == 255) {
if (j == 0) {	//top boundary
if (Dynamic_map_x_value[j*roirect.at(x).width + i - 1] > Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1])
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[j*roirect.at(x).width + i - 1];
else
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1];
}
else if (j == roirect.at(x).height-1) { //down  boundary
if (Dynamic_map_x_value[j*roirect.at(x).width + i - 1] > Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i-1])
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[j*roirect.at(x).width + i - 1];
else
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1];
}
else {	// middle
if (Dynamic_map_x_value[j*roirect.at(x).width + i - 1] < Dynamic_map_x_value[(j+1)*roirect.at(x).width + i-1])
if (Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1] < Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1])
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j - 1)*roirect.at(x).width + i - 1];
else
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j + 1)*roirect.at(x).width + i - 1];
else
if (Dynamic_map_x_value[(j-1)*roirect.at(x).width + i-1] > Dynamic_map_x_value[j*roirect.at(x).width + i - 1])
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[(j-1)*roirect.at(x).width + i - 1];
else
Dynamic_map_x_value[j*roirect.at(x).width + i] = ptrroi[j*roirect.at(x).width + i] + Dynamic_map_x_value[j*roirect.at(x).width + i - 1];
}

}
else
continue;

}
}
}
weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
//*****************show the weight map*********************
for (int y = 0; y < Dynamic_map_x_value.size(); y++) {

//weimap.create(mask.size(), CV_32F);
//weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
float * ptrweimap = weimap.ptr<float>(0);
ptrweimap[y] = Dynamic_map_x_value[y];
//imshow("weimap", weimap);
}
//*****************show the weight map*********************

if (roirect.at(x).height < 10 || (roirect.at(x).height == 0)) {	//condition to find seams
Dynamic_seam_sp.at(x+sp.size())=(Dynamic_seam_x);
//cout << "pushback null DSX  height: "<< roirect.at(x).height << endl;
}
else {
//from last col
int j;
for (j = 0; j < roirect.at(x).height ; j++) {
if (Dynamic_map_x_value[(roirect.at(x).width - 1) + Dynamic_seam_x.at(roirect.at(x).width - 1)*roirect.at(x).width] < Dynamic_map_x_value[(roirect.at(x).width - 1) + j*roirect.at(x).width]) {
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
if (Dynamic_map_x_value[j*roirect.at(x).width + i] <= Dynamic_map_x_value[(j+1)*roirect.at(x).width + i])
Dynamic_seam_x.at(i) = j + 1;
else
Dynamic_seam_x.at(i) = j;
}
//down boundary
else if (j == roirect.at(x).height - 1) {
if (Dynamic_map_x_value[j*roirect.at(x).width + i] <= Dynamic_map_x_value[(j-1)*roirect.at(x).width + i])
Dynamic_seam_x.at(i) = j - 1;
else
Dynamic_seam_x.at(i) = j;
}
//middle
//if((j!= roirect.at(x).width - 1)&&(j!=0)) {
else {
if (Dynamic_map_x_value[j*roirect.at(x).width + i] < Dynamic_map_x_value[(j+1)*roirect.at(x).width + i]) {
if (Dynamic_map_x_value[(j-1)*roirect.at(x).width + i] > Dynamic_map_x_value[(j+1)*roirect.at(x).width + i])
Dynamic_seam_x.at(i) = j - 1;
else
Dynamic_seam_x.at(i) = j + 1;
}
else {
if (Dynamic_map_x_value[(j-1)*roirect.at(x).width + i] > Dynamic_map_x_value[j*roirect.at(x).width + i])
Dynamic_seam_x.at(i) = j - 1;
else
Dynamic_seam_x.at(i) = j;
}
}

}
//cout << "x :" << x << endl;
Dynamic_seam_sp.at(x+sp.size())=(Dynamic_seam_x);
Dynamic_seam_x.clear();
Dynamic_map_x_value.clear();
}
}

//********************************x_axis********************************************************

//initial rst
rst.resize(4 * sp.size());

for (int x = 0; x < sp.size(); x++) {
for (int y = 0; y < sp.at(x).size(); y++) {
//cout << "x :" << x << "  y :" << y << endl;
col = (sp.at(x).at(y) % src.cols) - roirect[x].x;
row = (sp.at(x).at(y) / src.cols) - roirect[x].y;
if ((col <= Dynamic_seam_sp.at(x).at(row) ) && (row<=Dynamic_seam_sp.at(sp.size()+x).at(col))) {
//cout << "0  " << ((col <= Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
rst.at(4*x).push_back((((row+roirect[x].y)*src.cols))+(col + roirect[x].x));
}
else if ((col > Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) {
//cout << "1  " << ((col > Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
rst.at(4 * x+1).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
}
else if ((col <= Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) {
//cout << "2  " << ((col <= Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
rst.at(4 * x+2).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
}
else if ((col > Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) {
//cout << "3  " << ((col > Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
rst.at(4 * x+3).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
}
}
}







}
*/
// method 1 cause slope cut

//method 2 
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

void itersp(vector<vector<int>> sp, Mat & energy_y, Mat & energy_x, vector <vector<int>> &rst, vector <meta> &meta, vector <superpixel> &sup) {

	int rowindex;
	int colindex;
	int rowwidth;
	int colwidth;
	Mat roi;
	Mat roix;
	Mat mask;
	int col, row;
	uchar *ptrmask;
	float *ptrroi;
	float *ptrroix;
	vector <vector <int>*> colwidthvec;
	vector <float> Dynamic_map_y_value;
	vector <int> Dynamic_seam_y;
	vector <float> Dynamic_map_x_value;
	vector <int> Dynamic_seam_x;
	vector <vector<int>> Dynamic_seam_sp;


	for (int itr = 0; itr < Iter; itr++) {	//iter times
		for (int x = 0; x < sp.size(); x++) {
			if (sup.at(x).flag == true) {
				if (sup.at(x).var > 0.1 || sup.at(x).size >(SP*SP / (4) ^ (x + 1))) {
					// generate retangle***************************************************
					if (sp.at(x).size() == 0) {
						//continue;

						sup.at(x).rectangle = Rect(0, 0, 0, 0);
					}
					else {
						rowindex = sp.at(x).at(0) / src.cols;
						rowwidth = ((sp.at(x).at(sp.at(x).size() - 1)) / src.cols) + 1;
						for (int m = 0; m < rowwidth; m++) {
							colwidthvec.push_back(new vector<int>);
						}
						int temprowwidth = 0;
						int temprowwidthmin = src.cols;
						int temprowwidthmax = 0;
						int tempcolindex = src.cols;
						for (int y = 0; y < sp.at(x).size(); y++) {
							colwidthvec.at(sp.at(x).at(y) / src.cols)->push_back((int)sp.at(x).at(y));
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
						sup.at(x).rectangle = Rect(colindex, rowindex, colwidth, rowwidth);

					}
					// generate retangle***************************************************

					// generate mask roi **************************************************
					vector <float> weightmap_y;
					vector <float> weightmap_x;
					double center = gaussian(0, sup.at(x).rectangle.width);
					weightmap_y.resize(sup.at(x).rectangle.width);
					weightmap_x.resize(sup.at(x).rectangle.height);
					for (int i = -sup.at(x).rectangle.width / 2; i < sup.at(x).rectangle.width / 2; i++)
						weightmap_y.at(sup.at(x).rectangle.width / 2 + i) = gaussian(i, sup.at(x).rectangle.width) / center;
					for (int i = -sup.at(x).rectangle.height / 2; i < sup.at(x).rectangle.height / 2; i++)
						weightmap_x.at(sup.at(x).rectangle.height / 2 + i) = gaussian(i, sup.at(x).rectangle.height) / center;
					if (sup.at(x).size == 0) {
						continue;
					}
					else {
						energy_y(sup.at(x).rectangle).copyTo(roi);
						energy_x(sup.at(x).rectangle).copyTo(roix);
						mask = Mat::zeros(roi.size(), CV_8UC1);

						for (int y = 0; y < sp.at(x).size(); y++) {
							col = (sp.at(x).at(y) % src.cols) - sup.at(x).rectangle.x;
							row = (sp.at(x).at(y) / src.cols) - sup.at(x).rectangle.y;
							mask.at<uchar>(row, col) = 255;
						}
						roi.convertTo(roi, CV_32F);
						roix.convertTo(roix, CV_32F);
						ptrmask = mask.data;
						ptrroi = roi.ptr<float>();
						ptrroix = roix.ptr<float>();
					}
					Mat roivalue;
					Mat roivaluex;
					roivalue = Mat::zeros(roi.size(), CV_32F);
					roivaluex = Mat::zeros(roi.size(), CV_32F);

					for (int i = 0; i < sup.at(x).rectangle.height; i++) {
						for (int j = 0; j < sup.at(x).rectangle.width; j++) {
							if ((int)mask.at<uchar>(i, j) == 255) {
								roivalue.at<float>(i, j) = roi.at<float>(i, j);
								roivaluex.at<float>(i, j) = roix.at<float>(i, j);
							}
							else {
								roivalue.at<float>(i, j) = 0;
								roivaluex.at<float>(i, j) = 0;
							}
						}
					}
					sup.at(x).roiyvalue = roivalue;
					sup.at(x).roixvalue = roivaluex;

					// generate mask roi **************************************************

					// y dynamic ****************
					Dynamic_map_y_value.resize(sup.at(x).rectangle.height*sup.at(x).rectangle.width);
					Dynamic_seam_y.resize(sup.at(x).rectangle.height);
					for (int i = 0; i < sup.at(x).rectangle.height; i++) {
						for (int j = 0; j < sup.at(x).rectangle.width; j++) {
							ptrroi[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] * weightmap_y[j];
						}
					}

					for (int i = 0; i < sup.at(x).rectangle.height; i++) {
						for (int j = 0; j < sup.at(x).rectangle.width; j++) {

							//First row value from weight roi
							if (i == 0) {
								Dynamic_map_y_value[j] = ptrroi[j];
								//cout << "Dynamic_map_y_value[j]" << Dynamic_map_y_value[j] << endl;
							}
							//Following rows find the largest from above connected to itself
							else {
								if (j == 0) {	//left boundary
									if (Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j] > Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j + 1])
										Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j];
									else
										Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j + 1];
								}
								if (j == sup.at(x).rectangle.width) { //right  boundary
									if (Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j - 1] > Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j])
										Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j - 1];
									else
										Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j];
								}
								else {	// middle 
									if (Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j - 1] > Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j])
										if (Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j - 1] > Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j + 1])
											Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j - 1];
										else
											Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j + 1];
									else
										if (Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j] > Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j + 1])
											Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j];
										else
											Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] = ptrroi[i*sup.at(x).rectangle.width + j] + Dynamic_map_y_value[(i - 1)*sup.at(x).rectangle.width + j + 1];
								}
							}
						}
					}
					//from last row
					int j;
					for (j = 0; j < sup.at(x).rectangle.width; j++) {				//*************************-1???************************************
						if (Dynamic_map_y_value[(sup.at(x).rectangle.height - 1)*sup.at(x).rectangle.width + Dynamic_seam_y.at(sup.at(x).rectangle.height - 1)] < Dynamic_map_y_value[(sup.at(x).rectangle.height - 1)*sup.at(x).rectangle.width + j]) {
							//if (ptrmask[(sup.at(x).rectangle.height - 1)*sup.at(x).rectangle.width + j] == 255) {
							Dynamic_seam_y.at(sup.at(x).rectangle.height - 1) = j;
							//cout <<"last row j : " <<j << endl;
							//}
							//cout << "Dynamic_seam_y.at(sup.at(x).rectangle.height - 1)" << Dynamic_seam_y.at(sup.at(x).rectangle.height - 1) << endl;
						}
					}
					for (int i = sup.at(x).rectangle.height - 2; i >= 0; i--) {
						//cout << "i:" << i << endl;
						j = Dynamic_seam_y.at(i + 1);
						//cout << "j:" << j << endl;
						//left boundary
						if (j == 0) {
							if (Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] <= Dynamic_map_y_value[i*sup.at(x).rectangle.width + j + 1]) {
								//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
								Dynamic_seam_y.at(i) = j + 1;
								//else
								//continue;
							}
							else {
								//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
								Dynamic_seam_y.at(i) = j;
								//else
								//continue;
							}
						}
						//right boundary
						else if (j == sup.at(x).rectangle.width - 1) {
							if (Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] <= Dynamic_map_y_value[i*sup.at(x).rectangle.width + j - 1]) {
								//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
								Dynamic_seam_y.at(i) = j - 1;
								//else continue;
							}
							else {
								//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
								Dynamic_seam_y.at(i) = j;
								//else continue;
							}
						}
						//middle
						//if((j!= sup.at(x).rectangle.width - 1)&&(j!=0)) {
						else {
							if (Dynamic_map_y_value[i*sup.at(x).rectangle.width + j] < Dynamic_map_y_value[i*sup.at(x).rectangle.width + j + 1]) {
								if (Dynamic_map_y_value[i*sup.at(x).rectangle.width + j - 1] > Dynamic_map_y_value[i*sup.at(x).rectangle.width + j + 1]) {
									//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
									Dynamic_seam_y.at(i) = j - 1;
									//else continue;
								}
								else {
									//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
									Dynamic_seam_y.at(i) = j + 1;
									//else continue;
								}
							}
							else {
								if (Dynamic_map_y_value[i*sup.at(x).rectangle.width + j - 1] > Dynamic_map_y_value[i*sup.at(x).rectangle.width + j]) {
									//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
									Dynamic_seam_y.at(i) = j - 1;
									//else continue;
								}
								else {
									//if (ptrmask[i*sup.at(x).rectangle.width + j] == 255)
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
				else
					break;
			}
			else
				continue;

		}
	}
}

void vec2txt(vector<vector<int>> srcvec){
	Mat labelmap;
	labelmap.create(src.size(), CV_32F);
	int col, row;
	Mat labelmapp;
	labelmapp = Mat::zeros(src.rows, src.cols, CV_8UC3);
	vector<avg_color_> avgcolor;
	avgcolor.reserve(50000);
	avgcolor.resize(50000);  //initial to zero value
	Mat rst;
	rst = Mat::zeros(src.rows, src.cols, CV_8UC3);


	for (int i = 0; i < srcvec.size(); i++) {
		for (int j = 0; j < srcvec.at(i).size(); j++) {
			col = srcvec.at(i).at(j) % src.cols;
			row = srcvec.at(i).at(j) / src.cols;
			labelmap.at<float>(row, col) = i + 1;
		}
	}

	//*******************random color*********************
	Mat labelshow;
	normalize(labelmap, labelshow, 1, 0, NORM_MINMAX);
	imshow("labelshow", labelshow);
	vector<Vec3b>random_color;
	for (int rr = 0; rr < 500000; rr++)
		random_color.push_back(Vec3b(rand() % 255, rand() % 255, rand() % 255));


	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			labelmapp.at<Vec3b>(i*src.cols + j) = random_color[(int)labelmap.at<float>(i*src.cols + j) % 4];
			//labelmapp.at<Vec3b>(i*src.cols + j) = random_color[(int)labelmap.at<float>(i*src.cols + j)];
		}
	}

	imshow("randomcolor", labelmapp);


	//*******************random color*********************

	//******************avg color calculate***************

	int i, j;

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			avgcolor[labelmap.at<float>(i*src.cols + j)].cnt = avgcolor[labelmap.at<float>(i*src.cols + j)].cnt + 1;
			avgcolor[labelmap.at<float>(i*src.cols + j)].r = avgcolor[labelmap.at<float>(i*src.cols + j)].r + (int)src.at<Vec3b>(i, j)[2];
			avgcolor[labelmap.at<float>(i*src.cols + j)].g = avgcolor[labelmap.at<float>(i*src.cols + j)].g + (int)src.at<Vec3b>(i, j)[1];
			avgcolor[labelmap.at<float>(i*src.cols + j)].b = avgcolor[labelmap.at<float>(i*src.cols + j)].b + (int)src.at<Vec3b>(i, j)[0];
		}
	}

	for (i = 0; i < avgcolor.size(); i++) {
		if (avgcolor[i].cnt != 0) {
			avgcolor[i].r = avgcolor[i].r / avgcolor[i].cnt;
			avgcolor[i].g = avgcolor[i].g / avgcolor[i].cnt;
			avgcolor[i].b = avgcolor[i].b / avgcolor[i].cnt;
		}
	}
	//avgcolr dump
	for (i = 0; i < src.rows; i++)
		for (j = 0; j < src.cols; j++) {
			Vec3b pix;
			pix[2] = avgcolor[labelmap.at<float>(i*src.cols + j)].r;
			pix[1] = avgcolor[labelmap.at<float>(i*src.cols + j)].g;
			pix[0] = avgcolor[labelmap.at<float>(i*src.cols + j)].b;

			rst.at<Vec3b>(i, j) = pix;

		}

	//****calculate labels***
	char labelcal[20];
	int labelconter = 0;
	for (int i = 0; i < srcvec.size(); i++) {
		if (srcvec.at(i).size() != 0)
			labelconter++;
	}
	sprintf(labelcal, "%dSP", labelconter);
	//****calculate labels***
	glogalspcounter = labelconter;
	cout << "Total SP NUMBERS: " << labelconter << endl;
	imshow("rstt", rst);
	//******************avg color calculate***************




	//********************** write file********************

	fstream fp;
	fp.open(filename_label, ios::out);
	if (!fp) {
		cout << "Fail to open file: " << filename_label << endl;
	}

	Mat map;
	map = Mat::zeros(src.rows, src.cols, CV_32F);
	int zcnt = 0;
	for (int i = 0; i < srcvec.size(); i++) {
		for (int j = 0; j < srcvec.at(i).size(); j++) {
			if (srcvec.at(i).size() == 0)
				zcnt++;
			int col, row;
			col = srcvec.at(i).at(j) % src.cols;
			row = srcvec.at(i).at(j) / src.cols;
			map.at<float>(row, col) = i + 1 - zcnt;
		}
	}
	//imshow("map", map);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			fp << (int)map.at<float>(i, j) << " ";

		}
		fp << endl;
	}

	fp.close();

	//********************** write file********************
}
void firstseam(vector <vector<int>> &superpixels, Mat &energyx, Mat &energyy, vector <vector<int>> &rst, vector <meta> &metaout, vector<superpixel> &supout) {
	vector <Rect> sprecty;
	vector <Rect> sprectx;

	vector<avg_color_> avgcolor;
	avgcolor.reserve(5000);
	avgcolor.resize(5000);
	Mat tempcolor;
	tempcolor.create(src.size(), CV_32F);

	supout.reserve(10000);
	supout.resize(10000);


	Mat roi;
	//Mat mask;
	Mat weimap;
	//uchar *ptrmask;
	float *ptrroi;

	int col, row;
	vector <float> weightmap_y;				//weight
	vector <float> weightmap_x;				//weight
	vector <float> Dynamic_map_y_value;		//roi value
	vector <int> Dynamic_seam_y;			// seam position
	vector <float> Dynamic_map_x_value;		//roi value
	vector <int> Dynamic_seam_x;			//seam position
	vector <vector<int>> Dynamic_seam_sp;	// store rst




	Dynamic_seam_sp.resize(src.cols / SP + src.rows / SP);


	//****************y************************

	int y_block;
	y_block = src.cols / SP;

	int first_block;
	int final_block;
	first_block = (src.cols%SP) / 2;
	final_block = (src.cols%SP) / 2;


	if ((src.cols%SP) % 2 != 0)
		first_block++;
	else
		first_block = first_block;

	//***************************************************
	sprecty.push_back(Rect(0, 0, first_block + SP, src.rows));

	for (int i = 1; i < y_block - 1; i++) {
		sprecty.push_back(Rect(i*SP + first_block, 0, SP, src.rows));
	}
	if (final_block != 0)
		sprecty.push_back(Rect(src.cols - (final_block + SP), 0, final_block + SP, src.rows));
	else
		sprecty.push_back(Rect(src.cols - SP, 0, SP, src.rows));
	//**************************************************

	for (int x = 0; x < sprecty.size(); x++) {

		if ((sprecty.at(x).width) <= SP / 2)
			continue;

		else {

			double center = gaussian(0, sprecty.at(x).width);
			weightmap_y.resize(sprecty.at(x).width);
			for (int i = -sprecty.at(x).width / 2; i < sprecty.at(x).width / 2; i++)
				weightmap_y.at(sprecty.at(x).width / 2 + i) = gaussian(i, sprecty.at(x).width) / center;   // 改變weight_map值 用gaussian 

			energyy(sprecty[x]).copyTo(roi);
			//mask = Mat::ones(roi.size(), CV_8UC1);

			/*for (int y = 0; y < superpixels.at(x).size(); y++) {
			col = (superpixels.at(x).at(y) % src.cols) - sprecty[x].x;
			row = (superpixels.at(x).at(y) / src.cols) - sprecty[x].y;
			mask.at<uchar>(row, col) = 255;
			}*/
			roi.convertTo(roi, CV_32FC1);
			//ptrmask = mask.data;
			ptrroi = roi.ptr<float>();

			Dynamic_map_y_value.resize(sprecty.at(x).height*sprecty.at(x).width);
			Dynamic_seam_y.resize(sprecty.at(x).height);
			//*********************USE_GRID_WEIGHT**************************************
			for (int i = 0; i < sprecty.at(x).height; i++) {
				for (int j = 0; j < sprecty.at(x).width; j++) {
					ptrroi[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] *weightmap_y[j];
				}
			}
			//*********************USE_GRID_WEIGHT**************************************
			for (int i = 0; i < sprecty.at(x).height; i++) {
				for (int j = 0; j < sprecty.at(x).width; j++) {
					//First row value from weight roi
					if (i == 0) {
						Dynamic_map_y_value[j] = ptrroi[j];
						//cout << "Dynamic_map_y_value[j]" << Dynamic_map_y_value[j] << endl;
					}
					//Following rows find the largest from above connected to itself
					else {

						if (j == 0) {	//left boundary
							if (Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j] > Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1])
								Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j];
							else
								Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1];
						}
						if (j == sprecty.at(x).width) { //right  boundary
							if (Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1] > Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j])
								Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1];
							else
								Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j];
						}
						else {	// middle 
							if (Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1] > Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j]){}
								if (Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1] > Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1])
									Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j - 1];
								else
									Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1];
							else
								if (Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j] > Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1])
									Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j];
								else
									Dynamic_map_y_value[i*sprecty.at(x).width + j] = ptrroi[i*sprecty.at(x).width + j] + Dynamic_map_y_value[(i - 1)*sprecty.at(x).width + j + 1];
						}
					}
				}
			}

			/*
			weimap = Mat::zeros(sprecty.at(x).height, sprecty.at(x).width, CV_32F);
			//*****************show the weight map*********************
			for (int y = 0; y < Dynamic_map_y_value.size(); y++) {

			//weimap.create(mask.size(), CV_32F);
			//weimap = Mat::zeros(roirect.at(x).height, roirect.at(x).width, CV_32F);
			float * ptrweimap = weimap.ptr<float>(0);
			ptrweimap[y] = Dynamic_map_y_value[y];
			imshow("weimap", weimap);
			}
			//*****************show the weight map*********************
			*/


			//from last row
			int j;
			for (j = 0; j < sprecty.at(x).width; j++) {				//*************************-1???************************************
				if (Dynamic_map_y_value[(sprecty.at(x).height - 1)*sprecty.at(x).width + Dynamic_seam_y.at(sprecty.at(x).height - 1)] < Dynamic_map_y_value[(sprecty.at(x).height - 1)*sprecty.at(x).width + j]) {
					//if (ptrmask[(roirect.at(x).height - 1)*roirect.at(x).width + j] == 255) {
					Dynamic_seam_y.at(sprecty.at(x).height - 1) = j;
					//cout <<"last row j : " <<j << endl;
					//}
					//cout << "Dynamic_seam_y.at(roirect.at(x).height - 1)" << Dynamic_seam_y.at(roirect.at(x).height - 1) << endl;
				}
			}
			for (int i = sprecty.at(x).height - 2; i >= 0; i--) {
				//cout << "i:" << i << endl;
				j = Dynamic_seam_y.at(i + 1);
				//cout << "j:" << j << endl;
				//left boundary
				if (j == 0) {
					if (Dynamic_map_y_value[i*sprecty.at(x).width + j] <= Dynamic_map_y_value[i*sprecty.at(x).width + j + 1]) {
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
				else if (j == sprecty.at(x).width - 1) {
					if (Dynamic_map_y_value[i*sprecty.at(x).width + j] <= Dynamic_map_y_value[i*sprecty.at(x).width + j - 1]) {
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
					if (Dynamic_map_y_value[i*sprecty.at(x).width + j] < Dynamic_map_y_value[i*sprecty.at(x).width + j + 1]) {
						if (Dynamic_map_y_value[i*sprecty.at(x).width + j - 1] > Dynamic_map_y_value[i*sprecty.at(x).width + j + 1]) {
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
						if (Dynamic_map_y_value[i*sprecty.at(x).width + j - 1] > Dynamic_map_y_value[i*sprecty.at(x).width + j]) {
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


	//****************y************************



	//****************x************************
	int x_block;
	x_block = src.rows / SP;

	first_block = (src.rows%SP) / 2;
	final_block = (src.rows%SP) / 2;


	if ((src.rows%SP) % 2 != 0)
		first_block++;
	else
		first_block = first_block;

	//***************************************************
	sprectx.push_back(Rect(0, 0, src.cols, first_block + SP));

	for (int i = 1; i < x_block - 1; i++) {
		sprectx.push_back(Rect(0, i*SP + first_block, src.cols, SP));
	}
	if (final_block != 0)
		sprectx.push_back(Rect(0, src.rows - (final_block + SP), src.cols, final_block + SP));
	else
		sprectx.push_back(Rect(0, src.rows - SP, src.cols, SP));
	//**************************************************

	for (int x = 0; x < sprectx.size(); x++) {

		if ((sprectx.at(x).height) <= SP / 2)
			continue;
		else {
			double center = gaussian(0, sprectx.at(x).height);
			weightmap_x.resize(sprectx.at(x).height);
			for (int i = -sprectx.at(x).height / 2; i < sprectx.at(x).height / 2; i++)
				weightmap_x.at(sprectx.at(x).height / 2 + i) = gaussian(i, sprectx.at(x).height) / center;



			energyx(sprectx[x]).copyTo(roi);
			//mask = Mat::ones(roi.size(), CV_8UC1);


			roi.convertTo(roi, CV_32F);
			//ptrmask = mask.data;
			ptrroi = roi.ptr<float>();
		}

		Dynamic_map_x_value.resize(sprectx.at(x).height*sprectx.at(x).width);
		Dynamic_seam_x.resize(sprectx.at(x).width);
		//*********************USE_GRID_WEIGHT**************************************
		for (int i = 0; i < sprectx.at(x).height; i++) {
			for (int j = 0; j < sprectx.at(x).width; j++) {
				ptrroi[i*sprectx.at(x).width + j] = ptrroi[i*sprectx.at(x).width + j];// *weightmap_x[i];
			}
		}
		//*********************USE_GRID_WEIGHT**************************************
		for (int i = 0; i < sprectx.at(x).width; i++) {
			for (int j = 0; j < sprectx.at(x).height; j++) {

				//First col value from weight roi
				if (i == 0) {

					Dynamic_map_x_value[j*sprectx.at(x).width] = ptrroi[j*sprectx.at(x).width];
					//cout << "Dynamic_map_x_value[j*roirect.at(x).width]" << Dynamic_map_x_value[j*roirect.at(x).width] << endl;

				}
				//Following cols find the largest from above connected to itself
				else {
					if (j == 0) {	//top boundary
						if (Dynamic_map_x_value[j*sprectx.at(x).width + i - 1] > Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1])
							Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[j*sprectx.at(x).width + i - 1];
						else
							Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1];
					}
					else if (j == sprectx.at(x).height - 1) { //down  boundary
						if (Dynamic_map_x_value[j*sprectx.at(x).width + i - 1] > Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1])
							Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[j*sprectx.at(x).width + i - 1];
						else
							Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1];
					}
					else {	// middle 
						if (Dynamic_map_x_value[j*sprectx.at(x).width + i - 1] < Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1])
							if (Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1] < Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1])
								Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1];
							else
								Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i - 1];
						else
							if (Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1] > Dynamic_map_x_value[j*sprectx.at(x).width + i - 1])
								Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i - 1];
							else
								Dynamic_map_x_value[j*sprectx.at(x).width + i] = ptrroi[j*sprectx.at(x).width + i] + Dynamic_map_x_value[j*sprectx.at(x).width + i - 1];
					}
				}
			}
		}

		//from last col
		int j;
		for (j = 0; j < sprectx.at(x).height; j++) {
			if (Dynamic_map_x_value[(sprectx.at(x).width - 1) + Dynamic_seam_x.at(sprectx.at(x).width - 1)*sprectx.at(x).width] < Dynamic_map_x_value[(sprectx.at(x).width - 1) + j*sprectx.at(x).width]) {
				//if(ptrmask[j*roirect.at(x).width+ roirect.at(x).width]==255)
				Dynamic_seam_x.at(sprectx.at(x).width - 1) = j;
				//cout << "Dynamic_seam_x.at(roirect.at(x).width - 1)" << Dynamic_seam_x.at(roirect.at(x).width - 1) << endl;
			}
		}
		for (int i = sprectx.at(x).width - 2; i >= 0; i--) {
			//cout << "width i:" << i << endl;
			j = Dynamic_seam_x.at(i + 1);
			//cout << "height j:" << j << endl;
			//top boundary
			if (j == 0) {
				if (Dynamic_map_x_value[j*sprectx.at(x).width + i] <= Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i])
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					Dynamic_seam_x.at(i) = j + 1;
				//else continue;
				else
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					Dynamic_seam_x.at(i) = j;
				//else continue;
			}
			//down boundary
			else if (j == sprectx.at(x).height - 1) {
				if (Dynamic_map_x_value[j*sprectx.at(x).width + i] <= Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i])
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					Dynamic_seam_x.at(i) = j - 1;
				//else continue;
				else
					//if (ptrmask[j*sprectx.at(x).width + i] == 255)
					Dynamic_seam_x.at(i) = j;
				//else continue;
			}
			//middle
			//if((j!= sprectx.at(x).width - 1)&&(j!=0)) {
			else {
				if (Dynamic_map_x_value[j*sprectx.at(x).width + i] < Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i]) {
					if (Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i] > Dynamic_map_x_value[(j + 1)*sprectx.at(x).width + i])
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j - 1;
					//else continue;
					else
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j + 1;
					//else continue;
				}
				else {
					if (Dynamic_map_x_value[(j - 1)*sprectx.at(x).width + i] > Dynamic_map_x_value[j*sprectx.at(x).width + i])
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j - 1;
					//else continue;
					else
						//if (ptrmask[j*sprectx.at(x).width + i] == 255)
						Dynamic_seam_x.at(i) = j;
					//else continue;
				}
			}

		}
		//cout << "x :" << x << endl;
		Dynamic_seam_sp.at(x + src.cols / SP) = (Dynamic_seam_x);
		Dynamic_seam_x.clear();
		Dynamic_map_x_value.clear();

	}

	//****************x************************

	//initial rst
	metaout.resize((src.cols)*(src.rows));
	unsigned char *rgb = (unsigned char*)(src.data);


	rst.resize((src.cols / SP + 1)*(src.rows / SP + 1));
	int block_y_cnt = 0;
	int block_x_cnt = 0;


	for (int i = 0; i < src.rows; i++) {
		block_y_cnt = 0;
		//block_x_cnt = 0;
		for (int j = 0; j < src.cols; j++) {
			if (block_y_cnt == src.cols / SP)
				block_y_cnt = block_y_cnt;
			else {
				if (j >= Dynamic_seam_sp.at(block_y_cnt).at(i) + sprecty.at(block_y_cnt).x) {
					if (block_y_cnt < src.cols / SP)
						block_y_cnt = block_y_cnt + 1;
					else
						block_y_cnt = block_y_cnt;
				}
				else {
					block_y_cnt = block_y_cnt;
				}
			}

			if (block_x_cnt == src.rows / SP) {
				if (i < Dynamic_seam_sp.at((src.cols / SP) + block_x_cnt - 1).at(j) + sprectx.at(block_x_cnt - 1).y)
					block_x_cnt--;
				else
					block_x_cnt = block_x_cnt;
			}
			else if (block_x_cnt == 0) {
				if (i >= Dynamic_seam_sp.at((src.cols / SP) + block_x_cnt).at(j) + sprectx.at(block_x_cnt).y)
					block_x_cnt = block_x_cnt + 1;
				else
					block_x_cnt = block_x_cnt;
			}
			else {
				if (i >= Dynamic_seam_sp.at((src.cols / SP) + block_x_cnt).at(j) + sprectx.at(block_x_cnt).y)
					block_x_cnt = block_x_cnt + 1;
				else if (i < Dynamic_seam_sp.at((src.cols / SP) + block_x_cnt - 1).at(j) + sprectx.at(block_x_cnt - 1).y)
					block_x_cnt = block_x_cnt - 1;
				else
					block_x_cnt = block_x_cnt;
			}





			/*
			if (block_x_cnt == src.rows / SP) {
			block_x_cnt = block_x_cnt;
			}
			else {
			if (i >= Dynamic_seam_sp.at((src.cols / SP) + block_x_cnt).at(j) + sprectx.at(block_x_cnt).y) {
			if (block_x_cnt < src.rows / SP)
			block_x_cnt = block_x_cnt + 1;
			else
			block_x_cnt = block_x_cnt;
			}
			else if (i < Dynamic_seam_sp.at((src.cols / SP) + block_x_cnt).at(j) + sprectx.at(block_x_cnt).y) {
			if (block_x_cnt > 0)
			block_x_cnt = block_x_cnt - 1;
			else
			block_x_cnt = block_x_cnt;
			}
			else
			block_x_cnt = block_x_cnt;
			}
			*/

			//0 1 2 3 4 5 6 7	0  1  2  3  4  5  6  7
			//1					8  9  10 11 12 13 14 15
			//2
			//3
			//4

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
	/*
	for (int x = 0; x < sp.size(); x++) {
	for (int y = 0; y < sp.at(x).size(); y++) {
	//cout << "x :" << x << "  y :" << y << endl;
	col = (sp.at(x).at(y) % src.cols) - roirect[x].x;
	row = (sp.at(x).at(y) / src.cols) - roirect[x].y;
	if ((col <= Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) {
	//cout << "0  " << ((col <= Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
	rst.at(4 * x).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
	}
	else if ((col > Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) {
	//cout << "1  " << ((col > Dynamic_seam_sp.at(x).at(row)) && (row <= Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
	rst.at(4 * x + 1).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
	}
	else if ((col <= Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) {
	//cout << "2  " << ((col <= Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
	rst.at(4 * x + 2).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
	}
	else if ((col > Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) {
	//cout << "3  " << ((col > Dynamic_seam_sp.at(x).at(row)) && (row > Dynamic_seam_sp.at(sp.size() + x).at(col))) << endl;
	rst.at(4 * x + 3).push_back((((row + roirect[x].y)*src.cols)) + (col + roirect[x].x));
	}
	}
	}

	*/
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
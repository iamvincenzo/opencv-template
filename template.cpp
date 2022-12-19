// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/features2d.hpp>

// std
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <vector>

// Esercitazione 1: demosaicizzazione-downsample
#define RGGB_BGGR 0
#define GRBG_GBRG 1

// Esercitazione 1: demosaicizzazione-luminance
#define RGGB 0
#define BGGR 1
#define GRBG 2
#define GBRG 3

// Disparity Bertozzi
#define MIN_DISPARITY 0
#define MAX_DISPARITY 127

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
	int top_left_x;
	int top_left_y;
	int h;
	int w;
	int padding_size;
	int threshold;
	int k;
	float alpha;
	int minThreshold;
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

////////////////////////////////////////////
// ESERCITAZIONI

// Esercitazione 1: demosaicizzazione
void downsample(const cv::Mat &src, cv::Mat &out, const u_char pattern)
{
	// shit indices
	int k1 = 0, l1 = 0;
	int k2 = 0, l2 = 0;
	int sum = 0;
	int mean = 0;
	int stride = 2;
	int G = 1;

	switch (pattern)
	{
	case RGGB_BGGR:
		k1 = 0;
		l1 = 1;
		k2 = 1;
		l2 = 0;
		std::cout << "RGGB o BGGR" << std::endl;
		break;
	case GRBG_GBRG:
		k1 = 0;
		l1 = 0;
		k2 = 1;
		l2 = 1;
		std::cout << "GRBG o GBRG" << std::endl;
		break;
	}

	out = cv::Mat(src.rows / 2, src.cols / 2, CV_8UC1, cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			// (u * stride) + l1) --> stride di 2 per downsample + l1/k1 = cella con G
			// (v * stride) + k1) * src.cols --> stride di 2 per downsample + l2/k2 = cella con G
			// G * src.elemSize1() --> BGR = 012 --> prende il canale G
			sum += (int)src.data[(((u * stride) + l1) + (((v * stride) + k1) * src.cols)) * src.elemSize() + G * src.elemSize1()];
			sum += (int)src.data[(((u * stride) + l2) + (((v * stride) + k2) * src.cols)) * src.elemSize() + G * src.elemSize1()];

			mean = sum / 2;
			out.data[u + v * out.cols] = mean;
			sum = 0;
			mean = 0;
		}
	}

	return;
}

// Esercitazione 1: demosaicizzazione
void luminance(const cv::Mat &src, cv::Mat &out, const u_char pattern)
{
	// shift indices
	int kg1 = 0, lg1 = 0, kg2 = 0, lg2 = 0, kr = 0, lr = 0, kb = 0, lb = 0;
	int chanB = 0, chanG = 1, chanR = 2;
	int B = 0, G1 = 0, G2 = 0, R = 0;

	out = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			if ((pattern == RGGB && (v % 2 == 0 && u % 2 == 0)) ||
				(pattern == BGGR && (v % 2 != 0 && u % 2 != 0)) ||
				(pattern == GRBG && (v % 2 == 0 && u % 2 != 0)) ||
				(pattern == GBRG && (v % 2 != 0 && u % 2 == 0)))
			{
				// RGGB
				kg1 = 0;
				lg1 = 1;
				kg2 = 1;
				lg2 = 0;
				kr = 0;
				lr = 0;
				kb = 1;
				lb = 1;
			}

			else if ((pattern == RGGB && (v % 2 == 0 && u % 2 != 0)) ||
					 (pattern == BGGR && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == GRBG && (v % 2 == 0 && u % 2 == 0)) ||
					 (pattern == GBRG && (v % 2 != 0 && u % 2 != 0)))
			{
				// GRBG
				kg1 = 0;
				lg1 = 0;
				kg2 = 1;
				lg2 = 1;
				kr = 0;
				lr = 1;
				kb = 1;
				lb = 0;
			}

			else if ((pattern == RGGB && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == BGGR && (v % 2 == 0 && u % 2 != 0)) ||
					 (pattern == GRBG && (v % 2 != 0 && u % 2 != 0)) ||
					 (pattern == GBRG && (v % 2 == 0 && u % 2 == 0)))
			{
				// GBRG
				kg1 = 0;
				lg1 = 0;
				kg2 = 1;
				lg2 = 1;
				kr = 1;
				lr = 0;
				kb = 0;
				lb = 1;
			}

			else if ((pattern == RGGB && (v % 2 != 0 && u % 2 != 0)) ||
					 (pattern == BGGR && (v % 2 == 0 && u % 2 == 0)) ||
					 (pattern == GRBG && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == GBRG && (v % 2 == 0 && u % 2 != 0)))
			{
				// BGGR
				kg1 = 0;
				lg1 = 1;
				kg2 = 1;
				lg2 = 0;
				kr = 1;
				lr = 1;
				kb = 0;
				lb = 0;
			}

			B = (int)src.data[((u + lb) + ((v + kb) * src.cols)) * src.elemSize() + chanB * src.elemSize1()];
			R = (int)src.data[((u + lr) + ((v + kr) * src.cols)) * src.elemSize() + chanR * src.elemSize1()];
			G1 = (int)src.data[((u + lg1) + ((v + kg1) * src.cols)) * src.elemSize() + chanG * src.elemSize1()];
			G2 = (int)src.data[((u + lg2) + ((v + kg2) * src.cols)) * src.elemSize() + chanG * src.elemSize1()];

			out.data[u + v * out.cols] = R * 0.3 + (G1 + G2) * 0.59 / 2.0 + B * 0.11;
		}
	}

	return;
}

// Esercitazione 1: demosaicizzazione
void simple(const cv::Mat &src, cv::Mat &out, const u_char pattern)
{
	// shift indices
	int kg1 = 0, lg1 = 0, kg2 = 0, lg2 = 0, kr = 0, lr = 0, kb = 0, lb = 0;
	int chanB = 0, chanG = 1, chanR = 2;
	int B = 0, G1 = 0, G2 = 0, R = 0;

	out = cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			if ((pattern == RGGB && (v % 2 == 0 && u % 2 == 0)) ||
				(pattern == BGGR && (v % 2 != 0 && u % 2 != 0)) ||
				(pattern == GRBG && (v % 2 == 0 && u % 2 != 0)) ||
				(pattern == GBRG && (v % 2 != 0 && u % 2 == 0)))
			{
				// RGGB shift
				kg1 = 0;
				lg1 = 1;
				kg2 = 1;
				lg2 = 0;
				kr = 0;
				lr = 0;
				kb = 1;
				lb = 1;
			}

			else if ((pattern == RGGB && (v % 2 == 0 && u % 2 != 0)) ||
					 (pattern == BGGR && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == GRBG && (v % 2 == 0 && u % 2 == 0)) ||
					 (pattern == GBRG && (v % 2 != 0 && u % 2 != 0)))
			{
				// GRBG shift
				kg1 = 0;
				lg1 = 0;
				kg2 = 1;
				lg2 = 1;
				kr = 0;
				lr = 1;
				kb = 1;
				lb = 0;
			}

			else if ((pattern == RGGB && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == BGGR && (v % 2 == 0 && u % 2 != 0)) ||
					 (pattern == GRBG && (v % 2 != 0 && u % 2 != 0)) ||
					 (pattern == GBRG && (v % 2 == 0 && u % 2 == 0)))
			{
				// GBRG shift
				kg1 = 0;
				lg1 = 0;
				kg2 = 1;
				lg2 = 1;
				kr = 1;
				lr = 0;
				kb = 0;
				lb = 1;
			}

			else if ((pattern == RGGB && (v % 2 != 0 && u % 2 != 0)) ||
					 (pattern == BGGR && (v % 2 == 0 && u % 2 == 0)) ||
					 (pattern == GRBG && (v % 2 != 0 && u % 2 == 0)) ||
					 (pattern == GBRG && (v % 2 == 0 && u % 2 != 0)))
			{
				// BGGR shift
				kg1 = 0;
				lg1 = 1;
				kg2 = 1;
				lg2 = 0;
				kr = 1;
				lr = 1;
				kb = 0;
				lb = 0;
			}

			B = (int)src.data[((u + lb) + ((v + kb) * src.cols)) * src.elemSize() + chanB * src.elemSize1()];
			G1 = (int)src.data[((u + lg1) + ((v + kg1) * src.cols)) * src.elemSize() + chanG * src.elemSize1()];
			G2 = (int)src.data[((u + lg2) + ((v + kg2) * src.cols)) * src.elemSize() + chanG * src.elemSize1()];
			R = (int)src.data[((u + lr) + ((v + kr) * src.cols)) * src.elemSize() + chanR * src.elemSize1()];

			out.data[(u + v * out.cols) * out.elemSize() + chanB * out.elemSize1()] = B;
			out.data[(u + v * out.cols) * out.elemSize() + chanG * out.elemSize1()] = (G1 + G2) / 2;
			out.data[(u + v * out.cols) * out.elemSize() + chanR * out.elemSize1()] = R;
		}
	}

	return;
}

// Esercitazione 1a: convoluzione
bool checkOddKernel(const cv::Mat &krn)
{
	if (krn.cols % 2 != 0 && krn.rows % 2 != 0)
		return true;
	else
		return false;
}

// Esercitazione 1a: convoluzione + binarizzazione + Canny
void addZeroPadding(const cv::Mat &src, cv::Mat &padded, const int padH, const int padW, bool zeroPad = true)
{
	/**
	 * Per immagini binarie
	 *
	 * padded_height = (input height + padding height top + padding height bottom)
	 * padded_width = (input width + padding width right + padding width left)
	 */
	if (zeroPad)
		padded = cv::Mat(src.rows + 2 * padH, src.cols + 2 * padW, CV_8UC1, cv::Scalar(0));
	else
		padded = cv::Mat(src.rows + 2 * padH, src.cols + 2 * padW, CV_8UC1, cv::Scalar(255));

	for (int v = padH; v < padded.rows - padH; ++v)
	{
		for (int u = padW; u < padded.cols - padW; ++u)
		{
			padded.at<u_char>(v, u) = src.at<u_char>((v - padH), (u - padW));
			// padded.data[(u + v * padded.cols) * padded.elemSize()] = src.data[((u - padW) + (v - padH) * src.cols) * src.elemSize()];
		}
	}

	return;
}

// Esercitazione 1a: convoluzione
void myfilter2D(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stride = 1)
{
	if (!checkOddKernel(krn))
	{
		std::cout << "ERRORE: il kernel deve essere dispari e quadrato." << std::endl;

		return;
	}

	int padH = (krn.rows - 1) / 2;
	int padW = (krn.cols - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	/**
	 * output_height = (int) ((input height + padding height top + padding height bottom - kernel height) / (stride height) + 1)
	 * output_width = (int) ((input width + padding width right + padding width left - kernel width) / (stride width) + 1)
	 */
	out = cv::Mat((int)((src.rows + 2 * padH - krn.rows) / stride) + 1, (int)((src.cols + 2 * padW - krn.cols) / stride) + 1, CV_32SC1);

	float g_kl;
	float w_sum;

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			w_sum = 0.0;

			for (int k = 0; k < krn.rows; ++k)
			{
				for (int l = 0; l < krn.cols; ++l)
				{
					g_kl = krn.at<float>(k, l);
					w_sum += g_kl * (float)padded.at<u_char>((v * stride) + k, (u * stride) + l);

					// g_kl = ((float *)krn.data)[l + k * krn.cols];
					// w_sum += g_kl * padded.data[(((u * stride) + l) + (((v * stride) + k) * padded.cols)) * padded.elemSize()];
				}
			}

			out.at<int32_t>(v, u) = w_sum;
			// out.data[(u + v * out.cols) * out.elemSize()] = w_sum;
		}
	}

	return;
}

// Esercitazione 2: background-subtraction-1
void computeForegroundPrevFrame(const cv::Mat &prevI, const cv::Mat &currI, cv::Mat &out, int th)
{
	int diff;

	for (int i = 0; i < (int)(currI.rows * currI.cols * currI.elemSize()); ++i)
	{
		diff = abs(currI.data[i] - prevI.data[i]);

		if (diff > th)
			out.data[i] = 255;
		else
			out.data[i] = 0;
	}

	return;
}

// Esercitazione 2: background-subtraction-2
void computeForegroundRunAvg(const std::vector<cv::Mat> prevKFrames, const cv::Mat &currI, cv::Mat &out, int th)
{
	int k = prevKFrames.size();
	int sum, avg, diff;

	// per ogni pixel (di tutte le immagini nel vettore)
	for (int i = 0; i < (int)(currI.rows * currI.cols * currI.elemSize()); ++i)
	{
		sum = 0;

		// di ogni immagine
		for (int j = 0; j < (int)prevKFrames.size(); ++j)
		{
			sum += prevKFrames[j].data[i]; // si sommano i valori dei pixel
		}

		avg = sum / k; // si calcola la media della somma del pixel i-esimo di ogni immagine j-esima

		diff = abs(currI.data[i] - avg); // si calcola la differenza |I(i,j) - B(i,j)|

		if (diff > th)
			out.data[i] = 255;
		else
			out.data[i] = 0;
	}

	return;
}

// Esercitazione 2: background-subtraction-3
void computeForegroundExpRunAvg(const std::vector<cv::Mat> prevKFrames, const cv::Mat &currI, cv::Mat &out, int th, float alpha)
{
	int k = prevKFrames.size();
	int sum, avg, diff, bg;

	// per ogni pixel (di tutte le immagini nel vettore)
	for (int i = 0; i < (int)(currI.rows * currI.cols * currI.elemSize()); ++i)
	{
		sum = 0;

		// di ogni immagine
		for (int j = 0; j < (int)prevKFrames.size(); ++j)
		{
			sum += prevKFrames[j].data[i]; // si sommano i valori dei pixel
		}

		avg = sum / k; // si calcola la media della somma del pixel i-esimo di ogni immagine j-esima

		bg = alpha * avg + (1 - alpha) * prevKFrames.back().data[i]; // l'immagine In (ovvero il frame precedente) è l'ultimo elemento che è stato appena inserito nel vettore

		diff = abs(currI.data[i] - bg);

		if (diff > th)
			out.data[i] = 255;
		else
			out.data[i] = 0;
	}

	return;
}

// Esercitazione 3: binarizzazione
void generateHistogram(const cv::Mat &src, std::vector<int> &vecOfValues)
{
	for (int v = 0; v < src.rows; ++v)
	{
		for (int u = 0; u < src.cols; ++u)
		{
			// ++vecOfValues[(int)src.data[((u + v * src.cols) * src.elemSize())]];
			++vecOfValues[(int)src.at<u_char>(v, u)];
		}
	}

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void addZeroPaddingGeneral(const cv::Mat &src, const cv::Mat &krnl, cv::Mat &padded, const cv::Point anchor, bool zeroPad = true)
{
	int padHTop = anchor.x;
	int padHBottom = krnl.rows - anchor.x - 1;
	int padWLeft = anchor.y;
	int padWRight = krnl.cols - anchor.y - 1;

	if (zeroPad)
	{
		padded = cv::Mat(src.rows + padHTop + padHBottom, src.cols + padWLeft + padWRight, CV_8UC1, cv::Scalar(0));
	}
	else // non crea problemi nel caso dell'erosione
	{
		padded = cv::Mat(src.rows + padHTop + padHBottom, src.cols + padWLeft + padWRight, CV_8UC1, cv::Scalar(255));
	}

	for (int v = padHTop; v < padded.rows - padHBottom; ++v)
	{
		for (int u = padWLeft; u < padded.cols - padWRight; ++u)
		{
			padded.at<u_char>(v, u) = src.at<u_char>(v - padHTop, u - padWLeft);
		}
	}

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myErodeBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outErodeB, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor);

	outErodeB = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	bool diff;

	for (int v = 0; v < outErodeB.rows; ++v)
	{
		for (int u = 0; u < outErodeB.cols; ++u)
		{
			diff = false;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (krnl.data[j + i * krnl.cols] != padded.data[(u + i) + (v + j) * padded.cols])
						{
							diff = true;
							break;
						}
					}
				}

				if (diff)
					break;
			}

			if (!diff)
				outErodeB.data[(u + v * outErodeB.cols)] = 255;
		}
	}

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myDilateBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outDilateB, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor);

	outDilateB = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	bool eq;

	for (int v = 0; v < outDilateB.rows; ++v)
	{
		for (int u = 0; u < outDilateB.cols; ++u)
		{
			eq = false;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (krnl.data[j + i * krnl.cols] == padded.data[(u + i) + (v + j) * padded.cols])
						{
							eq = true;
							break;
						}
					}
				}

				if (eq)
					break;
			}

			if (eq)
				outDilateB.data[(u + v * outDilateB.cols)] = 255;
		}
	}

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myOpenBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outOpenB, const cv::Point anchor)
{
	cv::Mat tmp;
	myErodeBinary(src, krnl, tmp, anchor);

	myDilateBinary(tmp, krnl, outOpenB, anchor);

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myCloseBinary(cv::Mat &src, cv::Mat &krnl, cv::Mat &outCloseB, const cv::Point anchor)
{
	cv::Mat tmp;
	myDilateBinary(src, krnl, tmp, anchor);

	myErodeBinary(tmp, krnl, outCloseB, anchor);

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myErodeGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outErodeG, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor, false);

	outErodeG = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	int min;

	for (int v = 0; v < outErodeG.rows; ++v)
	{
		for (int u = 0; u < outErodeG.cols; ++u)
		{
			min = 255;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (padded.data[(u + i) + (v + j) * padded.cols] < min)
						{
							min = padded.data[(u + i) + (v + j) * padded.cols];
						}
					}
				}
			}

			outErodeG.data[(u + v * outErodeG.cols)] = min;
		}
	}

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myDilateGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outDilateG, const cv::Point anchor)
{
	cv::Mat padded;
	addZeroPaddingGeneral(src, krnl, padded, anchor);

	outDilateG = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	int max;

	for (int v = 0; v < outDilateG.rows; ++v)
	{
		for (int u = 0; u < outDilateG.cols; ++u)
		{
			max = 0;

			for (int i = 0; i < krnl.rows; ++i)
			{
				for (int j = 0; j < krnl.cols; ++j)
				{
					if (krnl.data[j + i * krnl.cols] == 255)
					{
						if (padded.data[(u + i) + (v + j) * padded.cols] > max)
						{
							max = padded.data[(u + i) + (v + j) * padded.cols];
						}
					}
				}
			}

			outDilateG.data[(u + v * outDilateG.cols)] = max;
		}
	}

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myOpenGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outOpenB, const cv::Point anchor)
{
	cv::Mat tmp;
	myErodeGrayScale(src, krnl, tmp, anchor);

	myDilateGrayScale(tmp, krnl, outOpenB, anchor);

	return;
}

// Esercitazione 3: erosione-dilatazione-apertura-chiusura
void myCloseGrayScale(cv::Mat &src, cv::Mat &krnl, cv::Mat &outCloseB, const cv::Point anchor)
{
	cv::Mat tmp;
	myDilateGrayScale(src, krnl, tmp, anchor);

	myErodeGrayScale(tmp, krnl, outCloseB, anchor);

	return;
}

/* Canny: Vincenzo */
// Esercitazione 4a: canny edge detector
void gaussianKrnl(float sigma, int r, cv::Mat &krnl)
{
	krnl = cv::Mat(2 * r + 1, 1, CV_32FC1, cv::Scalar(0.0));

	float sum = 0.0;

	std::cout << std::endl;

	// calcolo kernel - formula 1D: (1/((sqrt(CV_PI * 2)*sig)) * exp(-x^2) / (2 * sig^2))))
	for (int x = -r; x <= r; ++x)
	{
		krnl.at<float>(x + r, 0) = (exp(-pow(x, 2) / (2 * pow(sigma, 2)))) / (sqrt(CV_PI * 2) * sigma);

		// calcolo della somma dei pesi
		sum += krnl.at<float>(x + r, 0);
	}

	// normalizzazione del kernel
	krnl /= sum;

	// stampa kernel
	std::cout << "Vertical Gaussian Kernel - 1D:\n"
			  << std::endl;

	for (int v = 0; v < krnl.rows; ++v)
	{
		for (int u = 0; u < krnl.cols; ++u)
		{
			std::cout << krnl.at<float>(v, u) << "\t";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;

	// display kernel
	cv::namedWindow("gaussian krnl 1D - vertical", cv::WINDOW_NORMAL);
	cv::imshow("gaussian krnl 1D - vertical", krnl);

	return;
}

// Esercitazione 4a: canny edge detector
void GaussianBlur(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride = 1)
{
	// vertical gaussian filter creation
	cv::Mat gaussKrnl;
	gaussianKrnl(sigma, r, gaussKrnl);

	// horizontal gaussian filter creation
	cv::Mat gaussKrnlT;
	cv::transpose(gaussKrnl, gaussKrnlT);

	// display horizontal kernel
	cv::namedWindow("gaussian krnl 1D - horizontal", cv::WINDOW_NORMAL);
	cv::imshow("gaussian krnl 1D - horizontal", gaussKrnlT);

	// custom convolution
	cv::Mat myfilter2DresultTmp;
	myfilter2D(src, gaussKrnl, myfilter2DresultTmp, stride);

	// conversion intermediate result form CV_32SC1 --> CV_8UC1
	cv::Mat conversionTmp;
	myfilter2DresultTmp.convertTo(conversionTmp, CV_8UC1);

	// custom convolution
	cv::Mat outTmp;
	myfilter2D(conversionTmp, gaussKrnlT, outTmp, stride);
	outTmp.convertTo(out, CV_8UC1);

	return;
}

// Esercitazione 4a: canny edge detector
void sobel3x3(const cv::Mat &src, cv::Mat &magn, cv::Mat &orient)
{
	float dataKx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float dataKy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	cv::Mat Kx(3, 3, CV_32F, dataKx);
	cv::Mat Ky(3, 3, CV_32F, dataKy);

	cv::Mat Ix;
	myfilter2D(src, Kx, Ix);

	cv::Mat Iy;
	myfilter2D(src, Ky, Iy);

	// compute magnitude
	Ix.convertTo(Ix, CV_32F);
	Iy.convertTo(Iy, CV_32F);
	cv::pow(Ix.mul(Ix) + Iy.mul(Iy), 0.5, magn);

	// compute orientation
	orient = cv::Mat(Ix.size(), CV_32FC1);
	for (int v = 0; v < Ix.rows; ++v)
	{
		for (int u = 0; u < Ix.cols; ++u)
		{
			orient.at<float>(v, u) = atan2f(Iy.at<float>(v, u), Ix.at<float>(v, u));
		}
	}

	// scale on 0-255 range
	// per quanto riguarda Sobel, si sfrutta convertScaleAbs e non convertTo() perchè ci sono i valori negativi
	cv::Mat aIx, aIy, amagn;
	cv::convertScaleAbs(Ix, aIx);
	cv::convertScaleAbs(Iy, aIy);
	cv::convertScaleAbs(magn, amagn);

	// display vertical sobel
	cv::namedWindow("vertical sobel", cv::WINDOW_NORMAL);
	cv::imshow("vertical sobel", aIx);

	// display vertical sobel
	cv::namedWindow("horizontal sobel", cv::WINDOW_NORMAL);
	cv::imshow("horizontal sobel", aIy);

	// display sobel magnitude
	cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
	cv::imshow("sobel magnitude", amagn);

	// trick to display orientation
	cv::Mat adjMap;
	cv::convertScaleAbs(orient, adjMap, 255 / (2 * CV_PI));
	cv::Mat falseColorsMap;
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN); // COLORMAP_JET
	cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
	cv::imshow("sobel orientation", falseColorsMap);

	return;
}

// Esercitazione 4a: canny edge detector
template <class T>
float bilinear(const cv::Mat &src, float r, float c)
{
	// r in [0,rows-1] - c in [0,cols-1]
	if (r < 0 || r > (src.rows - 1) || c < 0 || c > (src.cols - 1))
		return -1;

	// get the largest possible integer less than or equal to r/c
	int rfloor = floor(r);
	int cfloor = floor(c);
	float t = r - rfloor;
	float s = c - cfloor;

	return (src.at<T>(rfloor, cfloor)) * (1 - s) * (1 - t) +
		   (src.at<T>(rfloor, cfloor + 1)) * s * (1 - t) +
		   (src.at<T>(rfloor + 1, cfloor)) * (1 - s) * t +
		   (src.at<T>(rfloor + 1, cfloor + 1)) * t * s;
}

// Esercitazione 4a: canny edge detector
int findPeaksBilInterpInterp(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{
	// Non Maximum Suppression

	out = cv::Mat(magn.rows, magn.cols, CV_32FC1, cv::Scalar(0.0));

	// convert orient from radiant to angles
	cv::Mat angles(orient.rows, orient.cols, orient.type(), cv::Scalar(0.0));
	orient.copyTo(angles);
	// angles *= (180 / CV_PI);

	float e1 = 0.0, e1x = 0.0, e1y = 0.0, e2 = 0.0, e2x = 0.0, e2y = 0.0;
	float theta = 0.0;

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int r = 1; r < angles.rows - 1; ++r)
	{
		for (int c = 1; c < angles.cols - 1; ++c)
		{
			theta = angles.at<float>(r, c);

			e1x = c + 1 * cos(theta);
			e1y = r + 1 * sin(theta);
			e2x = c - 1 * cos(theta);
			e2y = r - 1 * sin(theta);

			e1 = bilinear<float>(magn, e1y, e1x);
			e2 = bilinear<float>(magn, e2y, e2x);

			// magn.at<float>(r, c) is a local maxima
			if (magn.at<float>(r, c) >= e1 && magn.at<float>(r, c) >= e2)
			{
				out.at<float>(r, c) = magn.at<float>(r, c);
			}
		}
	}

	// scale on 0-255 range
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);

	// display sobel magnitude
	cv::namedWindow("sobel magnitude NMS - bilinInterp", cv::WINDOW_NORMAL);
	cv::imshow("sobel magnitude NMS - bilinInterp", outDisplay);

	return 0;
}

// Esercitazione 4a: canny edge detector
void findOptTreshs(const cv::Mat &src, float &tlow, float &thigh)
{
	float sum = 0.0;
	int N = 0;
	float medianPix = 0.0;

	for (int v = 0; v < src.rows; ++v)
	{
		for (int u = 0; u < src.cols; ++u)
		{
			sum += (float)src.at<u_char>(v, u);
			++N;
		}
	}

	medianPix = sum / N;

	// max(0, 0.7 * medianPix)
	if (0 > 0.7 * medianPix)
		tlow = 0;
	else
		tlow = 0.7 * medianPix;

	// min(255, 1.3 * medianPix)
	if (255 < 1.3 * medianPix)
		thigh = 255;
	else
		thigh = 1.3 * medianPix;

	std::cout << "\n(doubleTh) Optiaml tresholds: \ntlow: " << tlow << " - thigh: " << thigh << std::endl;

	return;
}

// Esercitazione 4a: canny edge detector
void findAdjRecursive(cv::Mat &out, const int r, const int c)
{
	// Adjacent pixel to pixel (i,j):
	// (i-1, j-1) - (i-1, j) - (i-1, j+1)
	// (i, j-1)   - (i, j)   - (i, j+1)
	// (i+1, j-1) - (i+1, j) - (i+1, j+1)

	for (int i = r - 1; i <= r + 1; ++i)
	{
		for (int j = c - 1; j <= c + 1; ++j)
		{
			// se il pixel ha una valore compreso tra T-low e T-High
			if (out.at<u_char>(i, j) != 0 && out.at<u_char>(i, j) != 255)
			{
				// diventa un pixel di bordo
				out.at<u_char>(i, j) = 255;
				// analisi ricorsiva dei suoi vicini in quanto pixel di bordo
				findAdjRecursive(out, i, j);
			}
		}
	}

	return;
}

// Esercitazione 4a: canny edge detector
void doubleThRecursive(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
{
	float tmpVal = 0.0;

	out = cv::Mat(magn.rows, magn.cols, magn.type(), cv::Scalar(0.0));

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			out.at<float>(v, u) = magn.at<float>(v, u);
		}
	}

	out.convertTo(out, CV_8UC1);

	// passata 1

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			tmpVal = magn.at<float>(v, u);

			// Over T-high: keep edge
			if (tmpVal >= t2)
			{
				out.at<u_char>(v, u) = 255;
			}
			// Under T-low: remove edge
			else if (tmpVal < t1)
			{
				out.at<u_char>(v, u) = 0;
			}
		}
	}

	// passata 2

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			// per ogni pixel di bordo avvia la procedura di crescita dei suoi vicini (ricorsiva)
			if (out.at<u_char>(v, u) == 255)
			{
				findAdjRecursive(out, v, u);
			}
		}
	}

	// passata 3: rimozione dei non massimi rimanenti

	for (int v = 1; v < out.rows - 1; ++v)
	{
		for (int u = 1; u < out.cols - 1; ++u)
		{
			if (out.at<u_char>(v, u) != 255)
			{
				out.at<u_char>(v, u) = 0;
			}
		}
	}

	return;
}

/************************************************************************************************/

////////////////////////////////////////////
// SVILUPPATE DALLA TEORIA

/* 03. Image-filtering */

// 2D-gaussian-filter
void gaussianKrnl2D(cv::Mat &krnl, float sigma, int r)
{
	krnl = cv::Mat(2 * r + 1, 2 * r + 1, CV_32FC1, cv::Scalar(0.0));

	float t = 0.0, sum = 0.0, s = 2 * sigma * sigma;

	// calcolo kernel
	for (int x = -r; x <= r; ++x)
	{
		for (int y = -r; y <= r; ++y)
		{
			t = sqrt((x * x) + (y * y));

			krnl.at<float>(x + r, y + r) = (exp(-(t * t) / s)) / (CV_PI * s);

			// calcolo della somma dei pesi
			sum += krnl.at<float>(x + r, y + r);
		}
	}

	// normalizzazione del kernel
	krnl /= sum;

	// stampa kernel
	std::cout << "Gaussian Kernel:\n"
			  << std::endl
			  << krnl
			  << std::endl;

	// display kernel
	cv::namedWindow("gaussian krnl 2D", cv::WINDOW_NORMAL);
	cv::imshow("gaussian krnl 2D", krnl);

	return;
}

// mean/box-filter
void myMeanBoxFilterSmoothing(const cv::Mat &src, cv::Mat &out, int stride = 1)
{
	int size = 3;
	cv::Mat boxKrnl(size, size, CV_32F, cv::Scalar(1.0 / 9));

	myfilter2D(src, boxKrnl, out, stride);

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myMeanBoxFilterSmoothing", cv::WINDOW_NORMAL);
	cv::imshow("myMeanBoxFilterSmoothing", outDisplay);
}

// shift-filter
void myShifted1px(const cv::Mat &src, cv::Mat &out, int stride = 1)
{
	int size = 3;
	cv::Mat boxKrnl(size, size, CV_32F, cv::Scalar(0));
	boxKrnl.at<float>(1, 2) = 1.0;

	myfilter2D(src, boxKrnl, out, stride);

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myShifted1px", cv::WINDOW_NORMAL);
	cv::imshow("myShifted1px", outDisplay);
}

// sharp filter
void mySharpeningFilter(const cv::Mat &src, cv::Mat &out, int stride = 1)
{
	int size = 3;
	cv::Mat krnl(size, size, CV_32F, cv::Scalar(0));
	krnl.at<float>(1, 1) = 2.0;
	cv::Mat boxKrnl(size, size, CV_32F, cv::Scalar(1.0 / 9));

	cv::Mat krnlSharp = krnl - boxKrnl;

	myfilter2D(src, boxKrnl, out, stride);

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("mySharpeningFilter", cv::WINDOW_NORMAL);
	cv::imshow("mySharpeningFilter", outDisplay);
}

// horizontal-gradient-filter
void myHorizontalGradient(const cv::Mat &src, cv::Mat &out, int stride = 1)
{
	int size = 3;
	float dataHG[9] = {0, 0, 0, -1, 0, 1, 0, 0, 0};
	cv::Mat Hgrad(size, size, CV_32F, dataHG);

	myfilter2D(src, Hgrad, out, stride);

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myHorizontalGradient", cv::WINDOW_NORMAL);
	cv::imshow("myHorizontalGradient", outDisplay);

	return;
}

// vertical-gradient-filter
void myVerticalGradient(const cv::Mat &src, cv::Mat &out, int stride = 1)
{
	int size = 3;
	float dataVG[9] = {0, 1, 0, 0, 0, 0, 0, -1, 0};
	cv::Mat Vgrad(size, size, CV_32F, dataVG);

	myfilter2D(src, Vgrad, out, stride);

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myVerticalGradient", cv::WINDOW_NORMAL);
	cv::imshow("myVerticalGradient", outDisplay);

	return;
}

// prewitt-filter
void myPrewitt(const cv::Mat &src, cv::Mat &magn, cv::Mat &orient)
{
	float dataKy[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
	float dataKx[9] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
	cv::Mat Kx(3, 3, CV_32F, dataKx);
	cv::Mat Ky(3, 3, CV_32F, dataKy);

	cv::Mat Ix;
	myfilter2D(src, Kx, Ix);

	cv::Mat Iy;
	myfilter2D(src, Ky, Iy);

	// compute magnitude
	Ix.convertTo(Ix, CV_32F);
	Iy.convertTo(Iy, CV_32F);
	cv::pow(Ix.mul(Ix) + Iy.mul(Iy), 0.5, magn);

	// compute orientation
	orient = cv::Mat(Ix.size(), CV_32FC1);
	for (int v = 0; v < Ix.rows; ++v)
	{
		for (int u = 0; u < Ix.cols; ++u)
		{
			orient.at<float>(v, u) = atan2f(Iy.at<float>(v, u), Ix.at<float>(v, u)) + 2 * CV_PI;
		}
	}

	// scale on 0-255 range
	// per quanto riguarda Sobel, si sfrutta convertScaleAbs e non convertTo() perchè ci sono i valori negativi
	cv::Mat aIx, aIy, amagn;
	cv::convertScaleAbs(Ix, aIx);
	cv::convertScaleAbs(Iy, aIy);
	cv::convertScaleAbs(magn, amagn);

	// display vertical sobel
	cv::namedWindow("vertical myPrewitt", cv::WINDOW_NORMAL);
	cv::imshow("vertical myPrewitt", aIx);

	// display vertical sobel
	cv::namedWindow("horizontal myPrewitt", cv::WINDOW_NORMAL);
	cv::imshow("horizontal myPrewitt", aIy);

	// display sobel magnitude
	cv::namedWindow("myPrewitt magnitude", cv::WINDOW_NORMAL);
	cv::imshow("myPrewitt magnitude", amagn);

	// trick to display orientation
	cv::Mat adjMap;
	cv::convertScaleAbs(orient, adjMap, 255 / (2 * CV_PI));
	cv::Mat falseColorsMap;
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);
	cv::namedWindow("myPrewitt orientation", cv::WINDOW_NORMAL);
	cv::imshow("myPrewitt orientation", falseColorsMap);

	return;
}

// log-filter
void myLoGFilter(const cv::Mat &src, cv::Mat &out, int stride = 1)
{
	int size = 3;
	float dataLog[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
	cv::Mat logFilter(size, size, CV_32F, dataLog);

	myfilter2D(src, logFilter, out, stride);

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myLoGFilter", cv::WINDOW_NORMAL);
	cv::imshow("myLoGFilter", outDisplay);
	return;
}

// median pooling
void medianPooling(const cv::Mat &src, const int krn_size, cv::Mat &out, int stride = 1)
{
	if (krn_size % 2 == 0)
	{
		std::cerr << "medianPooling() - ERROR: the kernel is not odd in size." << std::endl;
		exit(1);
	}

	if (src.type() != CV_8UC1)
	{
		std::cerr << "medianPooling() - ERROR: the source image is not uint8." << std::endl;
		exit(1);
	}

	int padH = (krn_size - 1) / 2;
	int padW = (krn_size - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	out = cv::Mat((int)((src.rows + 2 * padH - krn_size) / stride) + 1, (int)((src.cols + 2 * padW - krn_size) / stride) + 1, CV_32SC1);

	std::vector<int> values;

	for (int m = 0; m < out.rows; ++m)
	{
		for (int n = 0; n < out.cols; ++n)
		{
			values.clear();

			for (int k = 0; k < krn_size; ++k)
			{
				for (int l = 0; l < krn_size; ++l)
				{
					values.push_back(padded.at<u_char>((m * stride) + k, (n * stride) + l));
				}
			}

			sort(values.begin(), values.end());

			int index = values.size() / 2 - 1;

			// out.data[(n + m * out.cols) * out.elemSize()] = w_sum;
			out.at<int32_t>(m, n) = values[index];
		}
	}

	return;
}

// max-min pooling
void maxMinPooling(const cv::Mat &src, const int krn_size, cv::Mat &out, int stride = 1, const bool maxPooling = true)
{
	if (krn_size % 2 == 0)
	{
		std::cerr << "maxPooling() - ERROR: the kernel is not odd in size." << std::endl;
		exit(1);
	}

	if (src.type() != CV_8UC1)
	{
		std::cerr << "maxPooling() - ERROR: the source image is not uint8." << std::endl;
		exit(1);
	}

	int padH = (krn_size - 1) / 2;
	int padW = (krn_size - 1) / 2;

	cv::Mat padded;
	addZeroPadding(src, padded, padH, padW);

	out = cv::Mat((int)((src.rows + 2 * padH - krn_size) / stride) + 1, (int)((src.cols + 2 * padW - krn_size) / stride) + 1, CV_32SC1);

	int min_value = INT_MAX, max_value = INT_MIN;

	for (int m = 0; m < out.rows; ++m)
	{
		for (int n = 0; n < out.cols; ++n)
		{
			for (int k = 0; k < krn_size; ++k)
			{
				for (int l = 0; l < krn_size; ++l)
				{
					if (padded.at<u_char>((m * stride) + k, (n * stride) + l) < min_value)
						min_value = padded.at<u_char>((m * stride) + k, (n * stride) + l);

					if (padded.at<u_char>((m * stride) + k, (n * stride) + l) > max_value)
						max_value = padded.at<u_char>((m * stride) + k, (n * stride) + l);
				}
			}

			if (maxPooling)
				out.at<int32_t>(m, n) = max_value;

			else
				out.at<int32_t>(m, n) = min_value;
		}
	}

	return;
}

// bilateral-filter
void domainKrnl(const float sigma, const int r, cv::Mat &krnl)
{
	if (sigma <= 0)
	{
		std::cerr << "domainKrnl() - ERROR: the domain kernel sigma is not positive." << std::endl;
		exit(1);
	}

	if (r < 0)
	{
		std::cerr << "domainKrnl() - ERROR: kernel radius is negative." << std::endl;
		exit(1);
	}

	krnl = cv::Mat(2 * r + 1, 2 * r + 1, CV_32FC1);

	// float w_sum = 0.0f;  //sum of all values of the kernel (used for normalization)
	float weight;

	for (int v = 0; v < krnl.rows; ++v)
	{
		for (int u = 0; u < krnl.cols; ++u)
		{
			weight = exp(-((pow(v - (krnl.rows - r - 1), 2)) + pow(u - (krnl.cols - r - 1), 2)) / (2 * pow(sigma, 2)));
			krnl.at<float>(v, u) = weight;
		}
	}

	return;
}

// bilateral-filter
float distance(const int val1, const int val2)
{
	return pow(val1 - val2, 2);

	// alternativa: bilateral-filter
	// float distance(int x, int y, int i, int j)
	// {
	// 	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
	// }
}

// bilateral-filter
void rangeKrnl(const cv::Mat &ngb, const float sigma, const int radius, cv::Mat &krnl)
{
	if (sigma <= 0)
	{
		std::cerr << "rangeKrnl() - ERROR: the range kernel sigma is not positive." << std::endl;
		exit(1);
	}

	if (radius < 0)
	{
		std::cerr << "rangeKrnl() - ERROR: kernel radius is negative." << std::endl;
		exit(1);
	}

	krnl = cv::Mat(2 * radius + 1, 2 * radius + 1, CV_32FC1, cv::Scalar(0));

	float weight;
	int f_i_j = ngb.at<u_char>(radius, radius);

	int f_ik_jl;

	for (int r = 0; r < ngb.rows; ++r)
	{
		for (int c = 0; c < ngb.cols; ++c)
		{
			f_ik_jl = ngb.at<u_char>(r, c);
			// alternativa
			// weight = exp(-(pow(distance(x, y, x - c, y - r), 2)) / (2 * pow(sigma, 2)));
			weight = exp(-(pow(distance(f_i_j, f_ik_jl), 2)) / (2 * pow(sigma, 2)));
			krnl.at<float>(r, c) = weight;
		}
	}

	return;
}

// bilateral-filter
void myBilateralFilter(const cv::Mat &src, cv::Mat &out, const int radius, const float sigmaR, const float sigmaD)
{
	if (radius < 0)
	{
		std::cerr << "bilateralFilter() - ERROR: kernel radius is negative." << std::endl;
		exit(1);
	}

	cv::Mat dmnKrnl;

	// generate domain kernel
	domainKrnl(sigmaD, radius, dmnKrnl);

	out = cv::Mat((int)((src.rows + 2 * radius - dmnKrnl.rows) + 1), (int)((src.cols + 2 * radius - dmnKrnl.cols) + 1), CV_32SC1);

	cv::Mat padded;

	addZeroPadding(src, padded, radius, radius);

	cv::Mat rngKrnl, ngb, kernel;

	float w_sum; // weighted sum of the neighborhood pixels
	float g_kl;	 // value of the kernel at coordinates (k, l) (row, column)

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			w_sum = 0.0f;
			// neighborhood of current pixel (in the padded image)
			ngb = cv::Mat(padded, cv::Rect(c, r, dmnKrnl.cols, dmnKrnl.rows));

			rangeKrnl(ngb, sigmaR, radius, rngKrnl);

			kernel = dmnKrnl * rngKrnl;
			kernel /= cv::sum(kernel); // normalization

			for (int k = 0; k < kernel.rows; ++k)
			{
				for (int l = 0; l < kernel.cols; ++l)
				{
					g_kl = kernel.at<float>(k, l);

					w_sum += g_kl * (float)padded.at<u_char>(r + k, c + l);
				}
			}

			out.at<int32_t>(r, c) = w_sum;
		}
	}

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myBilateralFilter", cv::WINDOW_NORMAL);
	cv::imshow("myBilateralFilter", outDisplay);

	return;
}

// integral-image
void myIntegralImage(const cv::Mat &src, cv::Mat &out)
{
	src.copyTo(out);

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			for (int k = 0; k < out.channels(); ++k)
			{
				out.at<cv::Vec3b>(r, c)[k] = out.at<cv::Vec3b>(r, c)[k] + out.at<cv::Vec3b>(r, c - 1)[k] + out.at<cv::Vec3b>(r - 1, c)[k] - out.at<cv::Vec3b>(r - 1, c - 1)[k];
			}
		}
	}

	// usare convertScaleAbs quando nell'out ci sono dei valori negativi
	// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi

	// display custom convolution result
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	cv::namedWindow("myIntegralImage", cv::WINDOW_NORMAL);
	cv::imshow("myIntegralImage", outDisplay);

	return;
}

/* 04. Image-filtering 2 */

void generateHistogramGeneral(const cv::Mat &src, std::vector<std::vector<int>> &vecOfVec)
{
	if (src.type() == 16) // CV_8UC3
	{
		std::vector<int> vecOfValuesB;
		vecOfValuesB.resize(256);
		std::fill(vecOfValuesB.begin(), vecOfValuesB.end(), 0);

		std::vector<int> vecOfValuesG;
		vecOfValuesG.resize(256);
		std::fill(vecOfValuesG.begin(), vecOfValuesG.end(), 0);

		std::vector<int> vecOfValuesR;
		vecOfValuesR.resize(256);
		std::fill(vecOfValuesR.begin(), vecOfValuesR.end(), 0);

		for (int v = 0; v < src.rows; ++v)
		{
			for (int u = 0; u < src.cols; ++u)
			{
				for (int k = 0; k < src.channels(); ++k)
				{
					int index = (int)src.data[(u + v * src.cols) * src.elemSize() + k * src.elemSize1()];

					if (k == 0)
						++vecOfValuesB[index];
					else if (k == 1)
						++vecOfValuesG[index];
					else if (k == 2)
						++vecOfValuesR[index];
				}
			}
		}

		vecOfVec.push_back(vecOfValuesB);
		vecOfVec.push_back(vecOfValuesG);
		vecOfVec.push_back(vecOfValuesR);
	}

	else if (src.type() == 0) // CV_8UC1
	{
		std::vector<int> vecOfValues;
		vecOfValues.resize(256);
		std::fill(vecOfValues.begin(), vecOfValues.end(), 0);

		for (int v = 0; v < src.rows; ++v)
		{
			for (int u = 0; u < src.cols; ++u)
			{
				// ++vecOfValues[(int)src.data[((u + v * src.cols) * src.elemSize())]];
				++vecOfValues[(int)src.at<u_char>(v, u)];
			}
		}

		vecOfVec.push_back(vecOfValues);
	}

	else
	{
		std::cout << "Errore." << std::endl;
	}

	return;
}

// contrast-brightness
void myContrastBrightness(const cv::Mat &src, cv::Mat &out, const int a, const int b)
{
	out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			for (int k = 0; k < out.channels(); ++k)
			{
				out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] = a * (int)src.data[(u + v * src.cols) * src.elemSize() + k * src.elemSize1()] + b;
			}
		}
	}

	return;
}

// contrast-stretching
void myContrastStretching(const cv::Mat &src, cv::Mat &out)
{
	if (src.type() == 16) // CV_8UC3
	{
		float minfB = 100000, maxfB = 0, minfG = 100000, maxfG = 0, minfR = 100000, maxfR = 0;

		out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

		// get max e min in each channel
		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				for (int k = 0; k < out.channels(); ++k)
				{
					int val = (int)src.data[(u + v * src.cols) * src.elemSize() + k * src.elemSize1()];

					if (k == 0)
					{
						if (minfB > val)
							minfB = val;

						if (maxfB < val)
							maxfB = val;
					}

					else if (k == 1)
					{

						if (minfG > val)
							minfG = val;

						if (maxfG < val)
							maxfG = val;
					}

					else if (k == 2)
					{

						if (minfR > val)
							minfR = val;

						if (maxfR < val)
							maxfR = val;
					}
				}
			}
		}

		// std::cout << minfB << " " << maxfB << " " << minfG << " " << maxfG << " " << minfR << " " << maxfR << std::endl;

		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				for (int k = 0; k < out.channels(); ++k)
				{
					int val = (int)src.data[(u + v * src.cols) * src.elemSize() + k * src.elemSize1()];

					if (k == 0)
					{
						out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] = (u_char)255 * (val - minfB) / (maxfB - minfB);
					}

					if (k == 1)
					{
						out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] = (u_char)255 * (val - minfG) / (maxfG - minfG);
					}

					if (k == 2)
					{
						out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] = (u_char)255 * (val - minfR) / (maxfR - minfR);
					}

					// std::cout << "val: " << (int) out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] << std::endl;
				}
			}
		}
	}

	else if (src.type() == 0) // CV_8UC1
	{
		int minf = 100000, maxf = 0;

		out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				for (int k = 0; k < out.channels(); ++k)
				{
					int val = (int)src.data[(u + v * src.cols) * src.elemSize() + k * src.elemSize1()];

					if (minf > val)
						minf = val;

					if (maxf < val)
						maxf = val;
				}
			}
		}

		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				for (int k = 0; k < out.channels(); ++k)
				{
					int val = (int)src.data[(u + v * src.cols) * src.elemSize() + k * src.elemSize1()];

					out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] = (u_char)(val - minf) / (float)(maxf - minf);
				}
			}
		}
	}

	else
	{
		std::cout << "Errore." << std::endl;
	}

	return;
}

// linear-blend
void myLinearBlend(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &out, const int aplha)
{
	if (src1.type() == src2.type() && src1.rows == src2.rows && src1.cols == src2.cols) // CV_8UC3
	{
		out = cv::Mat(src1.rows, src1.cols, src1.type(), cv::Scalar(0));

		int val1 = 0, val2 = 0;

		// get max e min in each channel
		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				for (int k = 0; k < out.channels(); ++k)
				{
					val1 = (int)src1.data[(u + v * src1.cols) * src1.elemSize() + k * src1.elemSize1()];
					val2 = (int)src2.data[(u + v * src2.cols) * src2.elemSize() + k * src2.elemSize1()];

					out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] = (int)(((1 - aplha) * val1) + aplha * val2);
				}
			}
		}

		return;
	}
}

// gamma-correction
void myGammaCorrection(const cv::Mat &src, cv::Mat &out, const float lambda)
{
	out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

	int val = 0;

	// get max e min in each channel
	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			for (int k = 0; k < out.channels(); ++k)
			{
				val = (int)src.data[(u + v * src.cols) * src.elemSize() + k * src.elemSize1()];

				out.data[(u + v * out.cols) * out.elemSize() + k * out.elemSize1()] = pow(val, 1 / lambda);
			}
		}
	}

	return;
}

// histogram-equalization
void myHistogramEqualization(const cv::Mat &src, cv::Mat &out)
{
	// get image histogram
	std::vector<std::vector<int>> vecOfHisto;
	generateHistogramGeneral(src, vecOfHisto);

	out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

	int maxrange = 255;
	int index = 0;

	std::vector<float> probability(256, 0);
	std::vector<float> cumulativeProb(256, 0);
	std::vector<int> floorCumProb(256, 0);

	// get total amount of pixels
	float totPx = src.rows * src.cols;

	// foreach channel of the image
	for (int c = 0; c < src.channels(); ++c)
	{
		// calculate probability of each pixel intensity
		for (int i = 0; i < (int)vecOfHisto[c].size(); ++i)
		{
			probability[i] = vecOfHisto[c][i] / totPx;
		}

		cumulativeProb[0] = probability[0];

		// calculate the cumulative probability
		for (int i = 1; i < (int)cumulativeProb.size(); ++i)
		{
			cumulativeProb[i] = cumulativeProb[i - 1] + probability[i];
		}

		// change the intensity range
		for (int i = 0; i < (int)cumulativeProb.size(); ++i)
		{
			cumulativeProb[i] *= maxrange;
		}

		// round decimal values
		for (int i = 0; i < (int)floorCumProb.size(); ++i)
		{
			floorCumProb[i] = floor(cumulativeProb[i]);
		}

		// use the old pixel intensità as index of a lookup-table of new equalized values
		for (int v = 0; v < out.rows; ++v)
		{
			for (int u = 0; u < out.cols; ++u)
			{
				// out.at<u_char>(v, u) = (u_char)floorCumProb[(int)src.at<u_char>(v, u)];

				index = (int)src.data[(u + v * src.cols) * src.elemSize() + c * src.elemSize1()];
				out.data[(u + v * out.cols) * out.elemSize() + c * out.elemSize1()] = floorCumProb[index];
			}
		}
	}

	return;
}

/* 05. Binary vision */

// Metodo di Otsu: YouTube
void outsuMethod1(const cv::Mat &src, cv::Mat &out)
{
	//////////////////////
	// th - computation

	float sum_b = 0.0, sum_f = 0.0, wb = 0.0, wf = 0.0, tot = 0.0, ub = 0.0, uf = 0.0, sigma_b = 0.0;

	std::vector<float> thSigmaB;

	std::vector<int> vecOfValues;
	vecOfValues.resize(256);
	std::fill(vecOfValues.begin(), vecOfValues.end(), 0);

	generateHistogram(src, vecOfValues);

	for (int i = 0; i < (int)vecOfValues.size(); ++i)
	{
		tot += vecOfValues[i];
	}

	for (int th = 0; th < (int)vecOfValues.size(); ++th)
	{
		sum_b = 0.0;
		sum_f = 0.0;
		wb = 0.0;
		wf = 0.0;
		ub = 0.0;
		uf = 0.0;
		sigma_b = 0.0;

		for (int i = 0; i <= (int)th; ++i)
		{
			sum_b += vecOfValues[i];
			ub += (vecOfValues[i] * i);
		}

		if (sum_b == 0)
		{
			thSigmaB.push_back(0.0);
			continue;
		}

		for (int i = (int)th + 1; i <= 255; ++i)
		{
			sum_f += vecOfValues[i];
			uf += (vecOfValues[i] * i);
		}

		if (sum_f == 0)
		{
			thSigmaB.push_back(0.0);
			continue;
		}

		wb = sum_b / tot;
		ub = ub / sum_b;
		wf = sum_f / tot;
		uf = uf / sum_f;
		sigma_b = wb * wf * pow((ub - uf), 2);

		thSigmaB.push_back(sigma_b);
	}

	// indice del massimo nel vector
	int th = max_element(thSigmaB.begin(), thSigmaB.end()) - thSigmaB.begin();

	std::cout << "\n\noptimal-threshold: " << th << " and optimal-variance: " << thSigmaB[th] << std::endl;

	//////////////////////

	out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			if (src.data[((u + v * src.cols) * src.elemSize())] >= th)
			{
				out.data[((u + v * out.cols) * out.elemSize())] = 255;
			}
		}
	}

	return;
}

// Metodo di Otsu: Bertozzi
void outsuMethod2(const cv::Mat &src, cv::Mat &out)
{
	//////////////////////
	// th - computation

	float pBg = 0.0, pFg = 0.0, wBg = 0.0, wFg = 0.0, pAll = 0.0, varBg = 0.0, varFg = 0.0, varTot = 0.0,
		  sumBg = 0.0, sumFg = 0.0, meanXBg = 0.0, meanXFg = 0.0, sumVarBg = 0.0, sumVarFg = 0.0;

	std::vector<float> varTotVec;

	std::vector<int> vecOfValues;
	vecOfValues.resize(256);
	std::fill(vecOfValues.begin(), vecOfValues.end(), 0);

	generateHistogram(src, vecOfValues);

	for (int i = 0; i < (int)vecOfValues.size(); ++i)
	{
		pAll += vecOfValues[i]; // the total count of pixels in an image
	}

	for (int th = 0; th < (int)vecOfValues.size(); ++th)
	{
		pBg = 0.0;
		pFg = 0.0;
		wBg = 0.0;
		wFg = 0.0;
		varBg = 0.0;
		varFg = 0.0;
		varTot = 0.0;
		sumBg = 0.0;
		sumFg = 0.0;
		meanXBg = 0.0;
		meanXFg = 0.0;
		sumVarBg = 0.0;
		sumVarFg = 0.0;

		for (int i = 0; i <= (int)th; ++i)
		{
			pBg += vecOfValues[i]; // the count of background pixels at threshold th
			sumBg += (vecOfValues[i] * i);
		}

		if (pBg == 0)
		{
			varTotVec.push_back(0.0);
			continue;
		}

		meanXBg = sumBg / pBg;

		for (int i = (int)th + 1; i <= 255; ++i)
		{
			pFg += vecOfValues[i]; // the count of foreground pixels at threshold th
			sumFg += (vecOfValues[i] * i);
		}

		if (pFg == 0)
		{
			varTotVec.push_back(0.0);
			continue;
		}

		meanXFg = sumFg / pFg;

		wBg = pBg / pAll; // weight for background

		for (int i = 0; i <= (int)th; ++i)
		{
			sumVarBg += vecOfValues[i] * pow((i - meanXBg), 2);
		}

		varBg = sumVarBg / pBg; // (pBg - 1);

		wFg = pFg / pAll; // weight for foreground

		for (int i = (int)th + 1; i <= 255; ++i)
		{
			sumVarFg += vecOfValues[i] * pow((i - meanXFg), 2);
		}

		varFg = sumVarFg / pFg; // (pFg - 1);

		varTot = wBg * varBg + wFg * varFg;

		varTotVec.push_back(varTot);
	}

	// getMinIndexValue
	int th = 0;
	float minVal = 4000.0;

	for (int i = 0; i < (int)varTotVec.size(); ++i)
	{
		if (varTotVec[i] > 0 && varTotVec[i] < minVal)
		{
			minVal = varTotVec[i];
			th = i;
		}
	}

	std::cout << "\n\noptimal-threshold: " << th << " and optimal-variance: " << varTotVec[th] << std::endl;

	//////////////////////

	out = cv::Mat(src.rows, src.cols, src.type(), cv::Scalar(0));

	for (int v = 0; v < out.rows; ++v)
	{
		for (int u = 0; u < out.cols; ++u)
		{
			if (src.data[((u + v * src.cols) * src.elemSize())] >= th)
			{
				out.data[((u + v * out.cols) * out.elemSize())] = 255;
			}
		}
	}

	return;
}

// Metodo di Otsu: Checco
void adaptiveOtsuThresholding(const cv::Mat &src, const int minTh, cv::Mat &out)
{
	int best_threshold;
	double best_variance = std::numeric_limits<double>::max(); // we initialize the best variance with a very high value, because we want to minimize it
	double variance;

	double avg_below, avg_above; // respectively the arithmetic average of pixels below threshold and above threshold
	double var_below, var_above; // respectively the variance of the group of pixels below threshold and above threshold
	int sum_below, sum_above;	 // respectively the sum of the values of pixels below threshold and above threshold
	int n_below, n_above;		 // respectively the total number of pixels below threshold and above threshold

	// we start from a threshold of minTh in order to ignore too dark pixels
	for (int t = minTh; t < 255; ++t)
	{
		sum_below = 0;
		sum_above = 0;
		n_above = 0;
		n_below = 0;

		for (int i = 0; i < (int)(src.rows * src.cols * src.elemSize()); ++i)
		{
			if (src.data[i] >= t)
			{
				n_above++;
				sum_above += src.data[i];
			}
			else if (src.data[i] >= minTh)
			{
				n_below++;
				sum_below += src.data[i];
			}
		}

		if (n_below > 0)
			avg_below = sum_below / n_below;
		if (n_above > 0)
			avg_above = sum_above / n_above;

		var_below = 0;
		var_above = 0;

		for (int j = 0; j < (int)(src.rows * src.cols * src.elemSize()); ++j)
		{
			if (src.data[j] >= t)
				var_above += std::pow(src.data[j] - avg_above, 2);
			else if (src.data[j] >= minTh)
				var_below += std::pow(src.data[j] - avg_below, 2);
		}

		if (n_below > 0)
			var_below /= n_below;
		if (n_above > 0)
			var_above /= n_above;

		variance = var_below * n_below + var_above * n_above;

		if (variance < best_variance)
		{
			best_variance = variance;
			best_threshold = t;
		}
	}

	std::cout << "adaptiveOtsuThresholding(): best threshold is " << best_threshold << std::endl;

	out = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < (int)(src.rows * src.cols * src.elemSize()); ++i)
	{
		if (src.data[i] >= best_threshold)
		{
			out.data[i] = 255;
		}
	}
}

// Algoritmo row-by-row componenti connesse
void myRowByRowLabelling(const cv::Mat &bin, cv::Mat &labels)
{
	labels = cv::Mat(bin.rows, bin.cols, CV_8UC1, cv::Scalar(0));

	int pixRC, neighSx, neighUp, neighSxLabel, neighUpLabel, label = 0;

	std::vector<cv::Vec2i> equivalences; // first element is the bigger label value

	// passata 1
	for (int r = 0; r < bin.rows; ++r)
	{
		for (int c = 0; c < bin.cols; ++c)
		{
			pixRC = bin.at<u_char>(r, c);

			neighSx = -1;
			neighUp = -1;

			if (pixRC == 255)
			{
				if (c > 0 && bin.at<u_char>(r, c - 1) != 0)
				{
					neighSx = bin.at<u_char>(r, c - 1);
					neighSxLabel = labels.at<u_char>(r, c - 1);
				}
				if (r > 0 && bin.at<u_char>(r - 1, c) != 0)
				{
					neighUp = bin.at<u_char>(r - 1, c);
					neighUpLabel = labels.at<u_char>(r - 1, c);
				}

				if (neighSx == -1 && neighUp == -1)
					labels.at<u_char>(r, c) = ++label;

				else if (neighSx != -1 && neighUp == -1 && neighSx == pixRC)
					labels.at<u_char>(r, c) = labels.at<u_char>(r, c - 1);

				else if (neighSx != -1 && neighUp == -1 && neighSx != pixRC)
					labels.at<u_char>(r, c) = ++label;

				else if (neighUp != -1 && neighSx == -1 && neighUp == pixRC)
					labels.at<u_char>(r, c) = labels.at<u_char>(r - 1, c);

				else if (neighUp != -1 && neighSx == -1 && neighUp != pixRC)
					labels.at<u_char>(r, c) = ++label;

				else if (neighUp != -1 && neighSx != -1 && neighSx == pixRC && neighUp == pixRC && neighSxLabel == neighUpLabel)
					labels.at<u_char>(r, c) = neighSxLabel;

				else if (neighUp != -1 && neighSx != -1 && neighSx == pixRC && neighUp == pixRC && neighSxLabel != neighUpLabel)
				{
					if (neighSxLabel < neighUpLabel)
					{
						labels.at<u_char>(r, c) = neighSxLabel;
						equivalences.push_back(cv::Vec2i(neighUpLabel, neighSxLabel));
					}
					else
					{
						labels.at<u_char>(r, c) = neighUpLabel;
						equivalences.push_back(cv::Vec2i(neighSxLabel, neighUpLabel));
					}
				}
				else
					std::cout << "default" << std::endl;
			}
		}
	}

	// passata 2
	for (int r = 0; r < labels.rows; ++r)
	{
		for (int c = 0; c < labels.cols; ++c)
		{
			for (int i = 0; i < (int)equivalences.size(); ++i)
			{
				if (labels.at<u_char>(r, c) == equivalences[i][0])
				{
					labels.at<u_char>(r, c) = equivalences[i][1];
					break;
				}
			}
		}
	}

	return;
}

/* 06. Lines */

// Esercitazione 4a: canny edge detector
void findPeaks3x3(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{
	// Non Maximum Suppression
	// (i-1, j-1) - (i-1, j) - (i-1, j+1)
	// (i, j-1)   - (i, j)   - (i, j+1)
	// (i+1, j-1) - (i+1, j) - (i+1, j+1)

	out = cv::Mat(magn.rows, magn.cols, magn.type(), cv::Scalar(0.0));

	float e1 = 255.0, e2 = 255.0, theta = 0.0;

	// convert orient from radiant to angles
	cv::Mat angles(orient.rows, orient.cols, orient.type(), cv::Scalar(0.0));
	orient.copyTo(angles);
	angles *= (180 / CV_PI);

	for (int v = 0; v < angles.rows; ++v)
	{
		for (int u = 0; u < angles.cols; ++u)
		{
			// so that there aren't negative angles
			// all angles in range 180 (CV_PI = 180° is the atan2 periodicity)
			while (angles.at<float>(v, u) < 0)
			{
				angles.at<float>(v, u) += 180;
			}

			// all angles in range 180 (CV_PI = 180° is the atan2 periodicity)
			while (angles.at<float>(v, u) > 180)
			{
				angles.at<float>(v, u) -= 180;
			}
		}
	}

	// pixel di bordo scegliete voi la politica (ignorati --> ranges: [1, r-2] e [1, c-2])
	for (int v = 1; v < angles.rows - 1; ++v)
	{
		for (int u = 1; u < angles.cols - 1; ++u)
		{
			theta = angles.at<float>(v, u);

			// angle 0
			if ((0 <= theta && theta < 22.5) || (157.5 <= theta && theta <= 180))
			{
				e1 = magn.at<float>(v, u + 1);
				e2 = magn.at<float>(v, u - 1);
			}
			// angle 45
			else if (22.5 <= theta && theta < 67.5)
			{
				// gradient oblique direction
				e1 = magn.at<float>(v + 1, u - 1);
				e2 = magn.at<float>(v - 1, u + 1);
			}
			// angle 90
			else if (67.5 <= theta && theta < 112.5)
			{
				// gradient vertical direction
				e1 = magn.at<float>(v + 1, u);
				e2 = magn.at<float>(v - 1, u);
			}
			// angle 135
			else if (112.5 <= theta && theta < 157.5)
			{
				// gradient oblique direction
				e1 = magn.at<float>(v - 1, u - 1);
				e2 = magn.at<float>(v + 1, u + 1);
			}

			// magn.at<float>(r, c) is a local maxima
			if (magn.at<float>(v, u) >= e1 && magn.at<float>(v, u) >= e2)
			{
				out.at<float>(v, u) = magn.at<float>(v, u);
			}
		}
	}

	// scale on 0-255 range
	cv::Mat outDisplay;
	cv::convertScaleAbs(out, outDisplay);
	// in realtà, è possibile usare convertTo() perchè nella magnitude non ci sono valori negativi (per come è definita non possono esserci)
	out.convertTo(outDisplay, CV_8UC1);
	// display sobel magnitude
	cv::namedWindow("sobel magnitude NMS - 3x3mask", cv::WINDOW_NORMAL);
	cv::imshow("sobel magnitude NMS - 3x3mask", outDisplay);

	return;
}

// Trasformata di Hough: linee
void myPolarToCartesian(double rho, int theta, cv::Point &p1, cv::Point &p2, const int dist, const cv::Mat &img)
{
	if (theta >= 45 && theta <= 135)
	{
		// y = (r - x cos(t)) / sin(t)
		p1.x = 0;
		p1.y = ((double)(rho - (dist / 2)) - ((p1.x - (img.cols / 2)) * cos(theta * CV_PI / 180))) / sin(theta * CV_PI / 180) + (img.rows / 2);
		p2.x = img.cols;
		p2.y = ((double)(rho - (dist / 2)) - ((p2.x - (img.cols / 2)) * cos(theta * CV_PI / 180))) / sin(theta * CV_PI / 180) + (img.rows / 2);
	}
	else
	{
		// x = (r - y sin(t)) / cos(t);
		p1.y = 0;
		p1.x = ((double)(rho - (dist / 2)) - ((p1.y - (img.rows / 2)) * sin(theta * CV_PI / 180))) / cos(theta * CV_PI / 180) + (img.cols / 2);
		p2.y = img.rows;
		p2.x = ((double)(rho - (dist / 2)) - ((p2.y - (img.rows / 2)) * sin(theta * CV_PI / 180))) / cos(theta * CV_PI / 180) + (img.cols / 2);
	}

	return;
}

// Trasformata di Hough: linee
void myHoughTransfLines(const cv::Mat &image, cv::Mat &lines, const int minTheta, const int maxTheta, const int threshold)
{
	if (image.type() != CV_8UC1)
	{
		std::cerr << "houghLines() - ERROR: the image is not uint8." << std::endl;
		exit(1);
	}

	if (minTheta < 0 || minTheta >= maxTheta)
	{
		std::cerr << "houghLines() - ERROR: the minimum value of theta min_theta is out of the valid range [0, max_theta)." << std::endl;
		exit(1);
	}

	if (maxTheta <= minTheta || maxTheta > 180)
	{
		std::cerr << "houghLines() - ERROR: the maximum value of theta max_theta is out of the valid range (min_theta, PI]." << std::endl;
		exit(1);
	}

	int maxVal;

	if (image.rows > image.cols)
		maxVal = image.rows;
	else
		maxVal = image.cols;

	int max_distance = pow(2, 0.5) * maxVal / 2;

	std::vector<int> acc_row(maxTheta - minTheta + 1, 0);

	std::vector<std::vector<int>> accumulator(2 * max_distance, acc_row);

	for (int r = 0; r < image.rows; ++r)
	{
		for (int c = 0; c < image.cols; ++c)
		{
			if (image.at<u_char>(r, c) > 0)
			{
				for (int theta = minTheta; theta <= maxTheta; ++theta)
				{
					int rho = (c - image.cols / 2) * cos(theta * CV_PI / 180) + (r - image.rows / 2) * sin(theta * CV_PI / 180);

					++accumulator[rho + max_distance][theta];
				}
			}
		}
	}

	cv::Mat acc(2 * max_distance, maxTheta - minTheta, CV_8UC1);

	for (int r = 0; r < 2 * max_distance; ++r)
	{
		for (int t = minTheta; t <= maxTheta; ++t)
		{
			acc.at<u_char>(r, t) = accumulator[r][t];
		}
	}

	cv::namedWindow("Accumulator", cv::WINDOW_NORMAL);
	cv::imshow("Accumulator", acc);

	cv::Point start_point, end_point;

	for (int r = 0; r < acc.rows; ++r)
	{
		for (int t = minTheta; t < acc.cols; ++t)
		{
			if (accumulator[r][t] >= threshold)
			{
				myPolarToCartesian(r, t, start_point, end_point, acc.rows, image);

				cv::line(lines, start_point, end_point, cv::Scalar(0, 0, 255), 2, cv::LINE_4);

				std::cout
					<< "Start: (" << start_point.x << ", " << start_point.y << "); "
					<< "End: (" << end_point.x << ", " << end_point.y << ")"
					<< std::endl
					<< std::endl;
			}
		}
	}

	return;
}

/* // Trasformata di Hough: cerchi ???
void myHoughCircles(const cv::Mat &image, cv::Mat &circles, const int threshold)
{
	// non funzionante

	if (image.type() != CV_8UC1)
	{
		std::cerr << "houghCircles() - ERROR: the image is not uint8." << std::endl;
		exit(1);
	}

	// int max_cx = image.cols - 2;
	// int max_cy = image.rows - 2;
	// int max_r = 10;

	int accumulator[100][100][10] = {0};

	// for (int i = 0; i < 2; ++i)
	// {
	// 	for (int j = 0; j < 2; ++j)
	// 	{
	// 		for (int k = 0; k < 2; ++k)
	// 		{
	// 			std::cout << "i: " << i << " j:" << j << " k: " << k << " acc[i][j][k]: " << accumulator[i][j][k] << std::endl;
	// 		}
	// 	}
	// }

	// // std::vector<int> elem1(max_r, 0);
	// // std::vector<std::vector<int>> elem(max_cy, elem1);
	// // std::vector<std::vector<std::vector<int>>> accumulator(max_cx, elem);

	for (int r = 0; r < image.rows; ++r)
	{
		for (int c = 0; c < image.cols; ++c)
		{
			// check if the current pixel is an edge pixel
			if (image.at<uint8_t>(r, c) > 0)
			{
				int radius;
				// loop over all possible values of theta
				for (int i = 0; i < 100; ++i)
				{
					for (int j = 0; j < 100; ++j)
					{
						radius = pow(pow(c - i, 2) + pow(r - j, 2), 0.5);
						++accumulator[i][j][radius]; // increase the position in the accumulator
					}
				}
			}
		}
	}

	for (int i = 0; i < 100; ++i)
	{
		for (int j = 0; j < 100; ++j)
		{
			for (int r = 0; r < 10; ++r)
			{
				if (accumulator[i][j][r] >= threshold)
				{
					cv::Point center(i, j);
					// center
					cv::circle(circles, center, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
					// draw a red circle
					cv::circle(circles, center, r, cv::Scalar(0, 0, 255), 2, cv::LINE_4);
				}
			}
		}
	}

	return;
}*/

/* 09. Single-view-reconstruction */
	// Esercitazione 4/5 ???
	// Stima della geometria da un'immagine ???

/* 10. Epipolar-geometry */
	// rettificazione dell'immagine stereo ???

/* 11. Stereo-match */
	// find-best-match: cross-corr ???
	// find-best-match: cross-corr-norm ???
	// find-best-match: SAD-semi-incr ???
	// find-best-match: SAD-incr ???

/* 12. Model-fitting */
	// LS-line fitting ???
	// RANSAC ???

/* 13. Features-extraction */

// harris-corner-detection
void mySobelKrnls(cv::Mat &vSobel, cv::Mat &hSobel)
{
	vSobel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
			  -2, 0, 2,
			  -1, 0, 1);

	hSobel = (cv::Mat_<float>(3, 3) << 1, 2, 1,
			  0, 0, 0,
			  -1, -2, -1);

	return;
}

// harris-corner-detection
void myHarrisCornerDetector(const cv::Mat &src, std::vector<cv::KeyPoint> &key_points, const float harris_th, const float alpha)
{
	if (src.type() != CV_8UC1)
	{
		std::cerr << "harrisCornerDetector() - ERROR: the image is not uint8." << std::endl;
		exit(1);
	}

	cv::Mat hSobel, vSobel;

	mySobelKrnls(vSobel, hSobel);

	cv::Mat Ix, Iy;
	myfilter2D(src, vSobel, Ix);
	myfilter2D(src, hSobel, Iy);

	Ix.convertTo(Ix, CV_32F);
	Iy.convertTo(Iy, CV_32F);

	cv::Mat v_grad, h_grad;
	cv::convertScaleAbs(Ix, v_grad);
	cv::convertScaleAbs(Iy, h_grad);

	cv::Mat Ix_2 = cv::Mat(Ix.rows, Ix.cols, Ix.type()), Iy_2 = cv::Mat(Ix.rows, Ix.cols, Ix.type()), Ix_Iy = cv::Mat(Ix.rows, Ix.cols, Ix.type());

	for (int r = 0; r < Ix.rows; ++r)
	{
		for (int c = 0; c < Ix.cols; ++c)
		{
			Ix_2.at<float>(r, c) = Ix.at<float>(r, c) * Ix.at<float>(r, c);
			Iy_2.at<float>(r, c) = Iy.at<float>(r, c) * Iy.at<float>(r, c);
			Ix_Iy.at<float>(r, c) = Ix.at<float>(r, c) * Iy.at<float>(r, c);
		}
	}

	Ix_2.convertTo(Ix_2, CV_8UC1);
	Iy_2.convertTo(Iy_2, CV_8UC1);
	Ix_Iy.convertTo(Ix_Iy, CV_8UC1);

	float sigma = 20;
	int kRadius = 1;

	cv::Mat g_Ix_2, g_Iy_2, g_Ix_Iy;
	GaussianBlur(Ix_2, sigma, kRadius, g_Ix_2);
	GaussianBlur(Iy_2, sigma, kRadius, g_Iy_2);
	GaussianBlur(Ix_Iy, sigma, kRadius, g_Ix_Iy);

	g_Ix_2.convertTo(g_Ix_2, CV_32F);
	g_Iy_2.convertTo(g_Iy_2, CV_32F);
	g_Ix_Iy.convertTo(g_Ix_Iy, CV_32F);

	cv::Mat thetas(src.rows, src.cols, g_Ix_2.type(), cv::Scalar(0));

	for (int r = 0; r < thetas.rows; ++r)
	{
		for (int c = 0; c < thetas.cols; ++c)
		{
			float g_Ix_2_val = g_Ix_2.at<float>(r, c);
			float g_Iy_2_val = g_Iy_2.at<float>(r, c);
			float g_Ix_Iy_val = g_Ix_Iy.at<float>(r, c);

			float det = (g_Ix_2_val * g_Iy_2_val) - pow(g_Ix_Iy_val, 2);
			float trace = (g_Ix_2_val + g_Iy_2_val);
			thetas.at<float>(r, c) = det - alpha * pow(trace, 2);
		}
	}

	// (simplified) non-maxima suppression

	int ngbSize = 3;
	double valMax;
	float currTheta;

	for (int r = ngbSize / 2; r < thetas.rows - ngbSize / 2; ++r)
	{
		for (int c = ngbSize / 2; c < thetas.cols - ngbSize / 2; ++c)
		{
			currTheta = thetas.at<float>(r, c);

			if (currTheta <= harris_th)
				thetas.at<float>(r, c) = 0;

			if (currTheta > harris_th)
			{
				cv::Mat ngb(thetas, cv::Rect(c - ngbSize / 2, r - ngbSize / 2, ngbSize, ngbSize));

				cv::minMaxIdx(ngb, NULL, &valMax, NULL, NULL);

				// for (int i = 0; i < ngb.rows; ++i)
				// {
				// 	for (int j = 0; j < ngb.cols; ++j)
				// 	{
				// 		if (ngb.at<float>(r, c) > valMax)
				// 		{
				// 			valMax = ngb.at<float>(r, c);
				// 		}
				// 	}
				// }

				if (currTheta < valMax)
					thetas.at<float>(r, c) = 0;
			}
		}
	}

	// display response matrix
	cv::Mat adjMap, falseColorsMap;
	double minR, maxR;

	cv::minMaxLoc(thetas, &minR, &maxR);
	cv::convertScaleAbs(thetas, adjMap, 255 / (maxR - minR));
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

	// save keypoints
	for (int r = 0; r < thetas.rows; ++r)
	{
		for (int c = 0; c < thetas.cols; ++c)
		{
			if (thetas.at<float>(r, c) > 0)
				key_points.push_back(cv::KeyPoint(c, r, 5));
		}
	}

	return;
}

/* 14. Features-matching */

/************************************************************************************************/

////////////////////////////////////////////
// BERTOZZI

// Canny

void addPaddingProf(const cv::Mat image, cv::Mat &out, int vPadding, int hPadding)
{
	out = cv::Mat(image.rows + vPadding * 2, image.cols + hPadding * 2, image.type(), cv::Scalar(0));

	for (int row = vPadding; row < out.rows - vPadding; ++row)
	{
		for (int col = hPadding; col < out.cols - hPadding; ++col)
		{
			for (int k = 0; k < out.channels(); ++k)
			{
				out.data[((row * out.cols + col) * out.elemSize() + k * out.elemSize1())] = image.data[(((row - vPadding) * image.cols + col - hPadding) * image.elemSize() + k * image.elemSize1())];
			}
		}
	}

#if DEBUG
	std::cout << "Padded image " << out.rows << "x" << out.cols << std::endl;
	cv::namedWindow("Padded", cv::WINDOW_NORMAL);
	cv::imshow("Padded", out);
	unsigned char key = cv::waitKey(0);
#endif
}

void myfilter2DProf(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stridev, int strideh)
{
	if (!src.rows % 2 || !src.cols % 2)
	{
		std::cerr << "myfilter2D(): ERROR krn has not odd size!" << std::endl;
		exit(1);
	}

	int outsizey = (src.rows + (krn.rows / 2) * 2 - krn.rows) / (float)stridev + 1;
	int outsizex = (src.cols + (krn.cols / 2) * 2 - krn.cols) / (float)strideh + 1;
	out = cv::Mat(outsizey, outsizex, CV_32SC1);
	// std::cout << "Output image " << out.rows << "x" << out.cols << std::endl;

	cv::Mat image;
	addPaddingProf(src, image, krn.rows / 2, krn.cols / 2);

	int xc = krn.cols / 2;
	int yc = krn.rows / 2;

	int *outbuffer = (int *)out.data;
	float *kernel = (float *)krn.data;

	for (int i = 0; i < out.rows; ++i)
	{
		for (int j = 0; j < out.cols; ++j)
		{
			int origy = i * stridev + yc;
			int origx = j * strideh + xc;
			float sum = 0;
			for (int ki = -yc; ki <= yc; ++ki)
			{
				for (int kj = -xc; kj <= xc; ++kj)
				{
					sum += image.data[(origy + ki) * image.cols + (origx + kj)] * kernel[(ki + yc) * krn.cols + (kj + xc)];
				}
			}
			outbuffer[i * out.cols + j] = sum;
		}
	}
}

void gaussianKrnlProf(float sigma, int r, cv::Mat &krnl)
{
	float kernelSum = 0;
	krnl = cv::Mat(r * 2 + 1, 1, CV_32FC1);

	int yc = krnl.rows / 2;

	float sigma2 = pow(sigma, 2);

	for (int i = 0; i <= yc; i++)
	{
		int y2 = pow(i - yc, 2);
		float gaussValue = pow(M_E, -(y2) / (2 * sigma2));

		kernelSum += gaussValue;

		if (i != yc)
		{
			kernelSum += gaussValue;
		}

		((float *)krnl.data)[i] = gaussValue;
		((float *)krnl.data)[krnl.rows - i - 1] = gaussValue;
	}

	// Normalize.
	for (int i = 0; i < krnl.rows; i++)
	{
		((float *)krnl.data)[i] /= kernelSum;
	}
}

#define SEPARABLE
void GaussianBlurProf(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride)
{
	cv::Mat vg, hg;

	gaussianKrnlProf(sigma, r, vg);

#ifdef SEPARABLE
	hg = vg.t();
	std::cout << "DEBUG: Horizontal Gaussian Kernel:\n"
			  << hg << "\nSum: " << cv::sum(hg)[0] << std::endl;
	cv::Mat tmp;
	myfilter2DProf(src, hg, tmp, 1, stride);
	tmp.convertTo(tmp, CV_8UC1);
	myfilter2DProf(tmp, vg, out, stride, 1);
#else
	myfilter2DProf(src, vg * vg.t(), out, stride);
	std::cout << "DEBUG: Square Gaussian Kernel:\n"
			  << vg * vg.t() << "\nSum: " << cv::sum(vg * vg.t())[0] << std::endl;
#endif
}

void sobel3x3Prof(const cv::Mat &src, cv::Mat &magn, cv::Mat &ori)
{
	// SOBEL FILTERING
	// void cv::Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
	// sobel verticale come trasposto dell'orizzontale

	cv::Mat ix, iy;
	cv::Mat h_sobel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
					   -2, 0, 2,
					   -1, 0, 1);

	cv::Mat v_sobel = h_sobel.t();

	myfilter2DProf(src, h_sobel, ix, 1, 1);
	myfilter2DProf(src, v_sobel, iy, 1, 1);
	ix.convertTo(ix, CV_32FC1);
	iy.convertTo(iy, CV_32FC1);

	// compute magnitude
	cv::pow(ix.mul(ix) + iy.mul(iy), 0.5, magn);
	// compute orientation
	ori = cv::Mat(src.size(), CV_32FC1);
	float *dest = (float *)ori.data;
	float *srcx = (float *)ix.data;
	float *srcy = (float *)iy.data;

	for (int i = 0; i < ix.rows * ix.cols; ++i)
		dest[i] = atan2f(srcy[i], srcx[i]) + 2 * CV_PI;
}

template <typename T>
float bilinearProf(const cv::Mat &src, float r, float c)
{
	float yDist = r - int(r);
	float xDist = c - int(c);

	int value =
		src.at<T>(r, c) * (1 - yDist) * (1 - xDist) +
		src.at<T>(r + 1, c) * (yDist) * (1 - xDist) +
		src.at<T>(r, c + 1) * (1 - yDist) * (xDist) +
		src.at<T>(r + 1, c + 1) * yDist * xDist;

	return value;
}

int findPeaksProf(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{
	out = cv::Mat(magn.size(), magn.type(), cv::Scalar(0));

	for (int r = 1; r < magn.rows - 1; r++)
	{
		for (int c = 1; c < magn.cols - 1; c++)
		{

			float theta = orient.at<float>(r, c);
			float e1x = c + cos(theta);
			float e1y = r + sin(theta);
			float e2x = c - cos(theta);
			float e2y = r - sin(theta);

			float e1 = bilinearProf<float>(magn, e1y, e1x);
			float e2 = bilinearProf<float>(magn, e2y, e2x);
			float p = magn.at<float>(r, c);

			if (p < e1 || p < e2)
			{
				p = 0;
			}

			out.at<float>(r, c) = p;
		}
	}

	return 0;
}

int doubleThProf(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
{
	cv::Mat first = cv::Mat(magn.size(), CV_8UC1);

	float p; // little optimization (complier should cope with this)
	if (t1 >= t2)
		return 1;

	int tm = t1 + (t2 - t1) / 2;

	std::vector<cv::Point2i> strong;
	std::vector<cv::Point2i> low;
	for (int r = 0; r < magn.rows; r++)
	{
		for (int c = 0; c < magn.cols; c++)
		{
			if ((p = magn.at<float>(r, c)) >= t2)
			{
				first.at<uint8_t>(r, c) = 255;
				strong.push_back(cv::Point2i(c, r)); // BEWARE at<>() and point2i() use a different coords order...
			}
			else if (p <= t1)
			{
				first.at<uint8_t>(r, c) = 0;
			}
			else
			{
				first.at<uint8_t>(r, c) = tm;
				low.push_back(cv::Point2i(c, r));
			}
		}
	}

	first.copyTo(out);

	// grow points > t2
	while (!strong.empty())
	{
		cv::Point2i p = strong.back();
		strong.pop_back();
		// std::cout << p.y << " " << p.x << std::endl;
		for (int ox = -1; ox <= 1; ++ox)
			for (int oy = -1; oy <= 1; ++oy)
			{
				int nx = p.x + ox;
				int ny = p.y + oy;
				if (nx > 0 && nx < out.cols && ny > 0 && ny < out.rows && out.at<uint8_t>(ny, nx) == tm)
				{
					// std::cerr << ".";
					out.at<uint8_t>(ny, nx) = 255;
					strong.push_back(cv::Point2i(nx, ny));
				}
			}
	}

	// wipe out residual pixels < t2
	while (!low.empty())
	{
		cv::Point2i p = low.back();
		low.pop_back();
		if (out.at<uint8_t>(p.y, p.x) < 255)
			out.at<uint8_t>(p.y, p.x) = 0;
	}

	return 0;
}

// SAD

unsigned char openAndWait(const char *windowName, cv::Mat &image, const bool destroyWindow = true)
{
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::imshow(windowName, image);

	unsigned char key = cv::waitKey();

	if (key == 'q')
		exit(EXIT_SUCCESS);
	if (destroyWindow)
		cv::destroyWindow(windowName);

	return key;
}

/**
 * @brief This function is used to compute the disparity using the SAD approach.
 * (Note that it is neither the semi-incremental nor the full-incremental algorithm)
 *
 * @param left the left image [in]
 * @param right the right image [in]
 * @param w_size the size of the window (odd) [in]
 * @param out the output image [out]
 */
void SADDisparity(const cv::Mat &left, const cv::Mat &right, unsigned short w_size, cv::Mat &out)
{
	if (left.size() != right.size())
	{
		std::cerr << "SADDisparity() - ERROR: the left and right images has not the same size." << std::endl;
		exit(1);
	}

	if (w_size % 2 == 0)
	{
		std::cerr << "SADDisparity() - ERROR: the window is not odd in size." << std::endl;
		exit(1);
	}

	/*
	  The output is initialized with the same size of the left image (also right is good, because they have the same size) and it is a gray scale image.
	*/
	out = cv::Mat::zeros(left.size(), CV_8UC1);

	// with the first 2 for cycles we cycle on the rows and cols of the input images
	for (int r = w_size / 2; r < (left.rows - w_size / 2); ++r)
	{
		for (int c = w_size / 2; c < (left.cols - w_size / 2); ++c)
		{
			/*
			  For each point / pixel we compute the SAD and, in the end, we want the minimum value.
			  So we initialize minSAD (that contains each time the minimum computed SAD) with the highest possible number.
			*/
			unsigned int minSAD = UINT_MAX;
			int minSAD_d; // minSAD_d contains where (the disparity) we have found the minimum SAD

			// we compute all the possible disparities in the range [MIN_DISPARITY; MAX_DISPARITY] (in our case [0, 127])
			//(c - d) > 1 is needed to avoid exiting the image (we move at max for 127 positions or until we reach the end of the row of the image)
			for (int d = MIN_DISPARITY; d < MAX_DISPARITY && (c - d) > 1; ++d)
			{
				unsigned int SAD = 0; // the computed SAD

				// we cycle the w_size x w_size window (dr and dc are the offsets on the rows and cols with respect to the current pixel)
				for (int dr = -w_size / 2; dr <= w_size / 2; ++dr)
				{
					for (int dc = -w_size / 2; dc <= w_size / 2; ++dc)
					{
						int curr_r = r + dr;		   // the considered row according to the current element of the window
						int curr_left_c = c + dc;	   // the considered column (in the left image) according to the offset
						int curr_right_c = c - d + dc; // the considered column (in the right image) according to the offset
						SAD += abs(left.data[(curr_r * left.cols + curr_left_c) * left.elemSize1()] - right.data[(curr_r * right.cols + curr_right_c) * right.elemSize1()]);
					}
				}

				if (SAD < minSAD)
				{
					minSAD = SAD;
					minSAD_d = d;
				}
			}

			out.data[(r * left.cols + c) * out.elemSize1()] = minSAD_d;
		}
	}

	return;
}

/**
 * @brief This function is used to compute the disparity using the VDisparity approach.
 *
 * @param left the left image [in]
 * @param right the right image [in]
 * @param out the output image [out]
 */
void VDisparity(const cv::Mat &left, const cv::Mat &right, cv::Mat &out)
{
	out = cv::Mat::zeros(left.rows, MAX_DISPARITY, CV_8UC1);

	for (int r = 0; r < left.rows; ++r)
	{
		for (int c = 0; c < left.cols; ++c)
		{
			for (int d = 0; d < MAX_DISPARITY && (c - d) > 0; ++d)
			{

				if (left.data[(r * left.cols + c)] == right.data[(r * left.cols + c - d)])
				{
					out.data[(r * MAX_DISPARITY + d)] += 1;
				}
			}
		}
	}

	return;
}

/**
 * @brief This function is used to compute the disparity using the VDisparity approach.
 *
 * @param disparity the disparity image [in]
 * @param out the output image [out]
 */
void VDisparity(const cv::Mat &disparity, cv::Mat &out)
{
	out = cv::Mat::zeros(disparity.rows, MAX_DISPARITY, CV_8UC1); // the output image has the same number of rows of the disparity image and MAX_DISPARITY + 1 columns

	for (int r = 0; r < disparity.rows; ++r)
	{
		for (int c = 0; c < disparity.cols; ++c)
		{
			out.data[r * MAX_DISPARITY + disparity.data[r * disparity.cols + c]] += 1;
		}
	}

	return;
}

////////////////////////////////////////////

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int imreadflags = cv::IMREAD_COLOR;
	// int ksize = 3; // Esercitazione 1a: convoluzione
	// cv::Mat foreground, prevFrame;	  // Esercitazione 2: background-subtraction-1
	// std::vector<cv::Mat> lastKFrames; // Esercitazione 2: background-subtraction-2/3
	// cv::Mat foreground;				  // Esercitazione 2: background-subtraction-2/3

	//////////////////////
	// parse argument list:
	//////////////////////
	ArgumentList args;
	if (!ParseInputs(args, argc, argv))
	{
		exit(0);
	}

	while (!exit_loop)
	{
		// generating file name
		//
		// multi frame case
		if (args.image_name.find('%') != std::string::npos)
			sprintf(frame_name, (const char *)(args.image_name.c_str()), frame_number);
		else // single frame case
			sprintf(frame_name, "%s", args.image_name.c_str());

		// opening file
		std::cout << "Opening " << frame_name << std::endl;

		cv::Mat image = cv::imread(frame_name, imreadflags);
		if (image.empty())
		{
			std::cout << "Unable to open " << frame_name << std::endl;
			return 1;
		}

		std::cout << "The image has " << image.channels() << " channels, the size is " << image.rows << "x" << image.cols << " pixels "
				  << " the type is " << image.type() << " the pixel size is " << image.elemSize() << " and each channel is " << image.elemSize1() << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl;

		////////////////////////
		// ESERCITAZIONI

		/* // Esercizitazione 1: downsample
		{
			//////////////////////
			// processing code here

			std::string pattern = args.image_name.substr(21, 4);

			cv::Mat out;

			if ((pattern.compare("RGGB") == 0) || (pattern.compare("BGGR") == 0))
				downsample(image, out, RGGB_BGGR);

			else if ((pattern.compare("GRBG") == 0) || (pattern.compare("GBRG")) == 0)
				downsample(image, out, GRBG_GBRG);

			std::string title = "out " + pattern;

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image
			cv::namedWindow(title, cv::WINDOW_NORMAL);
			cv::imshow(title, out);

			// /////////////////////
		}*/

		/* // Esercitazione 1: luminance
		{
			//////////////////////
			// processing code here

			std::string pattern = args.image_name.substr(21, 4);

			cv::Mat out;

			if (pattern.compare("RGGB") == 0)
				luminance(image, out, RGGB);

			else if (pattern.compare("BGGR") == 0)
				luminance(image, out, BGGR);

			else if (pattern.compare("GRBG") == 0)
				luminance(image, out, GRBG);

			else if (pattern.compare("GBRG") == 0)
				luminance(image, out, GBRG);

			std::string title = "out " + pattern;

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image
			cv::namedWindow(title, cv::WINDOW_NORMAL);
			cv::imshow(title, out);

			/////////////////////
		}*/

		/* // Esercitazione 1: simple
		{
			//////////////////////
			// processing code here

			std::string pattern = args.image_name.substr(21, 4);

			cv::Mat out;

			if (pattern.compare("RGGB") == 0)
				simple(image, out, RGGB);

			else if (pattern.compare("BGGR") == 0)
				simple(image, out, BGGR);

			else if (pattern.compare("GRBG") == 0)
				simple(image, out, GRBG);

			else if (pattern.compare("GBRG") == 0)
				simple(image, out, GBRG);

			std::string title = "out " + pattern;

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image
			cv::namedWindow(title, cv::WINDOW_NORMAL);
			cv::imshow(title, out);

			/////////////////////
		}*/

		/* // Esercitazione 1a: convoluzione
		{
			//////////////////////
			// processing code here

			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			cv::Mat custom_kernel(ksize, ksize, CV_32FC1, 1.0 / (ksize * ksize));

			cv::Mat myfilter2Dresult;
			myfilter2D(grey, custom_kernel, myfilter2Dresult);

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image greyscale
			cv::namedWindow("grey", cv::WINDOW_NORMAL);
			cv::imshow("grey", grey);

			// display custom convolution result
			// usare convertScaleAbs quando nell'out ci sono dei valori negativi
			// usare out.convertTo(outDisplay, CV_8UC1) quando nell'out ci sono solo valori positivi
			cv::Mat outDisplay;
			cv::convertScaleAbs(myfilter2Dresult, outDisplay);
			cv::namedWindow("myfilter2D conv", cv::WINDOW_NORMAL);
			cv::imshow("myfilter2D conv", outDisplay);

			//////////////////////
		}*/

		/* // Esercitazione 2: background-subtraction-1
		{
			// uso: ./template -i ../Candela/Candela_m1.10_%06d.pgm -t 500

			// creazione del foreground
			foreground = cv::Mat(image.rows, image.cols, image.type());

			// se il frame è dal secondo in poi, è possibile calcoalre il foreground
			if (frame_number > 0)
				computeForegroundPrevFrame(prevFrame, image, foreground, args.threshold);

			// il background al passo n è il frame precedente
			image.copyTo(prevFrame);

			// display image
			cv::namedWindow("Original image", cv::WINDOW_NORMAL);
			cv::imshow("Original image", image);

			cv::namedWindow("Foreground", cv::WINDOW_NORMAL);
			cv::imshow("Foreground", foreground);
		}*/

		/* // Esercitazione 2: background-subtraction-2
		{
			// uso: ./template -i ../Candela/Candela_m1.10_%06d.pgm -t 500

			// creazione del foreground
			foreground = cv::Mat(image.rows, image.cols, image.type());

			// se all'interno del vettore sono presenti almeno k immagini
			if ((int)lastKFrames.size() == args.k)
				computeForegroundRunAvg(lastKFrames, image, foreground, args.threshold);

			// inserisci l'elemento all'interno del vettore (l'elemento viene posto alla fine)
			lastKFrames.push_back(image);

			// se all'interno del vettore ci sono più di k elementi, si rimuove il primo elemento (quello più vecchio)
			if ((int)lastKFrames.size() > args.k)
				lastKFrames.erase(lastKFrames.begin());

			// display image
			cv::namedWindow("Original image", cv::WINDOW_NORMAL);
			cv::imshow("Original image", image);

			cv::namedWindow("Foreground", cv::WINDOW_NORMAL);
			cv::imshow("Foreground", foreground);
		}*/

		/* // Esercitazione 2: background-subtraction-3
		{
			// uso: ./template -i ../Candela/Candela_m1.10_%06d.pgm -t 500

			// creazione del foreground
			foreground = cv::Mat(image.rows, image.cols, image.type());

			// se all'interno del vettore sono presenti almeno k immagini
			if ((int)lastKFrames.size() == args.k)
				computeForegroundExpRunAvg(lastKFrames, image, foreground, args.threshold, args.alpha);

			// inserisci l'elemento all'interno del vettore (l'elemento viene posto alla fine)
			lastKFrames.push_back(image);

			// se all'interno del vettore ci sono più di k elementi, si rimuove il primo elemento (quello più vecchio)
			if ((int)lastKFrames.size() > args.k)
				lastKFrames.erase(lastKFrames.begin());

			// display image
			cv::namedWindow("Original image", cv::WINDOW_NORMAL);
			cv::imshow("Original image", image);

			cv::namedWindow("Foreground", cv::WINDOW_NORMAL);
			cv::imshow("Foreground", foreground);
		}*/

		/* // Esercitazione 3: binarizzazione
		{
			//////////////////////
			// processing code here

			// create kernel (a croce)
			u_char dataKrnl[9] = {0, 255, 0, 255, 255, 255, 0, 255, 0};
			cv::Point anchor(1, 1);
			cv::Mat krnl(3, 3, CV_8UC1, dataKrnl);

			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			// Binary
			cv::Mat binaryImage;
			cv::threshold(grey, binaryImage, 150, 255, cv::THRESH_BINARY);

			cv::Mat outErodeB;
			myErodeBinary(binaryImage, krnl, outErodeB, anchor);

			cv::Mat outDilateB;
			myDilateBinary(binaryImage, krnl, outDilateB, anchor);

			cv::Mat outOpenB;
			myOpenBinary(binaryImage, krnl, outOpenB, anchor);

			cv::Mat outCloseB;
			myCloseBinary(binaryImage, krnl, outCloseB, anchor);

			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			cv::namedWindow("binary image", cv::WINDOW_NORMAL);
			imshow("binary image", binaryImage);

			cv::namedWindow("erosion binary", cv::WINDOW_NORMAL);
			imshow("erosion binary", outErodeB);

			cv::namedWindow("dilate binary", cv::WINDOW_NORMAL);
			imshow("dilate binary", outDilateB);

			cv::namedWindow("open binary", cv::WINDOW_NORMAL);
			imshow("open binary", outOpenB);

			cv::namedWindow("close binary", cv::WINDOW_NORMAL);
			imshow("close binary", outCloseB);

			// Gray scale

			cv::Mat outErodeG;
			myErodeGrayScale(grey, krnl, outErodeG, anchor);

			cv::Mat outDilateG;
			myDilateGrayScale(grey, krnl, outDilateG, anchor);

			cv::Mat outOpenG;
			myOpenGrayScale(grey, krnl, outOpenG, anchor);

			cv::Mat outCloseG;
			myCloseGrayScale(grey, krnl, outCloseG, anchor);

			cv::namedWindow("erosion grayscale", cv::WINDOW_NORMAL);
			imshow("erosion grayscale", outErodeG);

			cv::namedWindow("dilate grayscale", cv::WINDOW_NORMAL);
			imshow("dilate grayscale", outDilateG);

			cv::namedWindow("open grayscale", cv::WINDOW_NORMAL);
			imshow("open grayscale", outOpenG);

			cv::namedWindow("close grayscale", cv::WINDOW_NORMAL);
			imshow("close grayscale", outCloseG);

			/////////////////////
		}*/

		/* // Esercitazione 3: binarizzazione-Otsu
		{
			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			cv::Mat out1;
			outsuMethod1(grey, out1);

			cv::Mat out2;
			outsuMethod2(grey, out2);

			cv::Mat binarized;
			adaptiveOtsuThresholding(grey, args.minThreshold, binarized);

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image
			cv::namedWindow("grey image", cv::WINDOW_NORMAL);
			cv::imshow("grey image", grey);

			// display image
			cv::namedWindow("binary out1", cv::WINDOW_NORMAL);
			cv::imshow("binary out1", out1);

			// display image
			cv::namedWindow("binary out2", cv::WINDOW_NORMAL);
			cv::imshow("binary out2", out2);

			// display image
			cv::namedWindow("binarized", cv::WINDOW_NORMAL);
			cv::imshow("binarized", binarized);
		}*/

		/*// Esercitazione 4a: canny edge detector vincenzo
		{
			//////////////////////
			// processing code here

			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image greyscale
			cv::namedWindow("grey", cv::WINDOW_NORMAL);
			cv::imshow("grey", grey);

			// gaussian smoothing
			cv::Mat smoothGrey;
			GaussianBlur(grey, 1, 1, smoothGrey, 1);

			// sobel filtering
			cv::Mat magn;
			cv::Mat orient;
			sobel3x3(smoothGrey, magn, orient);

			cv::Mat outNms;
			// findPeaksBilInterpInterp(magn, orient, outNms);
			findPeaks3x3(magn, orient, outNms);

			float tlow, thigh;
			findOptTreshs(smoothGrey, tlow, thigh);

			cv::Mat outThr;
			doubleThRecursive(outNms, outThr, tlow, thigh);

			// display image greyscale
			cv::namedWindow("canny final result", cv::WINDOW_NORMAL);
			cv::imshow("canny final result", outThr);

			/////////////////////
		}*/

		/*// Esercitazione 4a: canny edge detector prof
		{
			int ksize = 3;
			int stride = 1;
			float sigma = 1.0f;

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// PROCESSING

			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			cv::Mat blurred, blurdisplay;
			GaussianBlurProf(grey, sigma, ksize / 2, blurred, stride);

			blurred.convertTo(blurdisplay, CV_8UC1);

			cv::namedWindow("Gaussian", cv::WINDOW_NORMAL);
			cv::imshow("Gaussian", blurdisplay);

			cv::Mat magnitude, orientation;
			sobel3x3Prof(blurdisplay, magnitude, orientation);

			cv::Mat magndisplay;
			magnitude.convertTo(magndisplay, CV_8UC1);
			cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
			cv::imshow("sobel magnitude", magndisplay);

			cv::Mat ordisplay;
			orientation.copyTo(ordisplay);
			float *orp = (float *)ordisplay.data;
			for (int i = 0; i < ordisplay.cols * ordisplay.rows; ++i)
				if (magndisplay.data[i] < 50)
					orp[i] = 0;
			cv::convertScaleAbs(ordisplay, ordisplay, 255 / (2 * CV_PI));
			cv::Mat falseColorsMap;
			cv::applyColorMap(ordisplay, falseColorsMap, cv::COLORMAP_JET);
			cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
			cv::imshow("sobel orientation", falseColorsMap);

			cv::Mat nms, nmsdisplay;
			findPeaksProf(magnitude, orientation, nms);
			nms.convertTo(nmsdisplay, CV_8UC1);
			cv::namedWindow("edges after NMS", cv::WINDOW_NORMAL);
			cv::imshow("edges after NMS", nmsdisplay);

			cv::Mat canny;
			if (doubleThProf(nms, canny, 50, 150))
			{
				std::cerr << "ERROR: t_low shoudl be lower than t_high" << std::endl;
				exit(1);
			}
			cv::namedWindow("Canny final result", cv::WINDOW_NORMAL);
			cv::imshow("Canny final result", canny);
		}*/

		////////////////////////

		/****************************************************************************************************************/

		////////////////////////
		// TEORIA

		/* // 03. Image filtering
		{
			//////////////////////
			// processing code here

			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image greyscale
			cv::namedWindow("grey", cv::WINDOW_NORMAL);
			cv::imshow("grey", grey);

			// 2D gaussian filter
			cv::Mat krnl2D;
			gaussianKrnl2D(krnl2D, 1, 1);

			// mean/box smoothing
			cv::Mat smoothBox;
			myMeanBoxFilterSmoothing(grey, smoothBox, 1);

			// shift 1 px
			cv::Mat shifted;
			myShifted1px(grey, shifted, 1);

			// sharped
			cv::Mat sharped;
			mySharpeningFilter(grey, sharped, 1);

			// sobel filtering
			cv::Mat magn;
			cv::Mat orient;
			sobel3x3(grey, magn, orient);

			// Hgradient
			cv::Mat Hgradient;
			myHorizontalGradient(grey, Hgradient, 1);

			// Vgradient
			cv::Mat Vgradient;
			myVerticalGradient(grey, Vgradient, 1);

			// gaussian smoothing
			cv::Mat smoothGrey;
			GaussianBlur(grey, 1, 1, smoothGrey, 1);

			// prewitt filtering
			cv::Mat magnPrewitt;
			cv::Mat orientPrewitt;
			myPrewitt(grey, magnPrewitt, orientPrewitt);

			// log filtering
			cv::Mat outLOG;
			myLoGFilter(grey, outLOG, 1);

			cv::Mat outBF;
			myBilateralFilter(grey, outBF, 1, 5, 2);

			cv::Mat outII;
			myIntegralImage(image, outII);

			/////////////////////
		}*/

		/* // 04. Image filtering 2
		{
			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			// display image
			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			// display image greyscale
			cv::namedWindow("grey", cv::WINDOW_NORMAL);
			cv::imshow("grey", grey);

			cv::Mat outCB;
			myContrastBrightness(image, outCB, 1.1, 16);

			cv::Mat outCS;
			myContrastStretching(image, outCS);

			cv::Mat outLB;
			cv::Mat src2 = cv::imread("../images/jenn.jpg", imreadflags);
			myLinearBlend(image, src2, outLB, 2);

			cv::Mat outCGC;
			myGammaCorrection(image, outCGC, 1.1);

			cv::Mat outHE;
			myHistogramEqualization(image, outHE);

			// display image
			cv::namedWindow("myContrastBrightness", cv::WINDOW_NORMAL);
			cv::imshow("myContrastBrightness", outCB);

			// display image
			cv::namedWindow("myContrastStretching", cv::WINDOW_NORMAL);
			cv::imshow("myContrastStretching", outCS);

			// display image
			cv::namedWindow("myLinearBlend", cv::WINDOW_NORMAL);
			cv::imshow("myLinearBlend", outLB);

			// display image
			cv::namedWindow("myGammaCorrection", cv::WINDOW_NORMAL);
			cv::imshow("myGammaCorrection", outCGC);

			// display image
			cv::namedWindow("myHistogramEqualization", cv::WINDOW_NORMAL);
			cv::imshow("myHistogramEqualization", outHE);
		}*/

		/* // 05. Binary vision: connected components
		{
			// create kernel (a croce)
			u_char dataKrnl[9] = {0, 255, 0, 255, 255, 255, 0, 255, 0};
			// u_char dataKrnl[9] = {255, 255, 255, 255, 255, 255, 255, 255, 255};
			cv::Point anchor(1, 1);
			cv::Mat krnl(3, 3, CV_8UC1, dataKrnl);

			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			// Binary
			cv::Mat binaryImage;
			cv::threshold(grey, binaryImage, 128, 255, cv::THRESH_BINARY);

			cv::Mat outOpen;
			myOpenBinary(binaryImage, krnl, outOpen, anchor);

			cv::Mat outClose;
			myCloseBinary(outOpen, krnl, outClose, anchor);

			cv::namedWindow("original image", cv::WINDOW_NORMAL);
			cv::imshow("original image", image);

			cv::namedWindow("binary image", cv::WINDOW_NORMAL);
			imshow("binary image", binaryImage);

			cv::namedWindow("open binary", cv::WINDOW_NORMAL);
			imshow("open binary", outOpen);

			cv::namedWindow("close binary", cv::WINDOW_NORMAL);
			imshow("close binary", outClose);

			// u_char data1[64] = {255, 255, 0, 255, 255, 255, 0, 255,
			// 					255, 255, 0, 255, 0, 255, 0, 255,
			// 					255, 255, 255, 255, 0, 0, 0, 255,
			// 					0, 0, 0, 0, 0, 0, 0, 255,
			// 					255, 255, 255, 255, 0, 255, 0, 255,
			// 					0, 0, 0, 255, 0, 255, 0, 255,
			// 					255, 255, 255, 255, 0, 0, 0, 255,
			// 					255, 255, 255, 255, 0, 255, 255, 255};

			// cv::Mat img(8, 8, CV_8UC1, data1);

			cv::Mat labels;
			myRowByRowLabelling(outClose, labels); // img

			// std::cout << labels << std::endl;

			cv::Mat connComp;
			cv::normalize(labels, connComp, 0, 255, cv::NORM_MINMAX);

			cv::namedWindow("connComp", cv::WINDOW_NORMAL);
			cv::imshow("connComp", connComp);
		}*/

		/*// 06. Lines: Trasformata di Hough-linee
		{
			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			cv::Mat blurred;
			cv::blur(grey, blurred, cv::Size(3, 3));

			cv::Mat contours;
			cv::Canny(blurred, contours, 50, 150, 3);

			cv::Mat lines;
			image.copyTo(lines);
			myHoughTransfLines(contours, lines, 0, 180, 250);

			// display image
			cv::namedWindow("image", cv::WINDOW_NORMAL);
			cv::imshow("image", image);

			// display image
			cv::namedWindow("contours", cv::WINDOW_NORMAL);
			cv::imshow("contours", contours);

			// display image
			cv::namedWindow("lines", cv::WINDOW_NORMAL);
			cv::imshow("lines", lines);
		}*/

		/* // 06. Lines: Trasformata di Hough-cerchi
		{
			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			cv::Mat blurred;
			blur(grey, blurred, cv::Size(3, 3));

			cv::Mat contours;
			cv::Canny(blurred, contours, 50, 200, 3);

			cv::Mat circles;
			image.copyTo(circles);
			myHoughCircles(contours, circles, 150);

			// display image
			cv::namedWindow("image", cv::WINDOW_NORMAL);
			cv::imshow("image", image);

			// display image
			cv::namedWindow("contours", cv::WINDOW_NORMAL);
			cv::imshow("contours", contours);

			// display image
			cv::namedWindow("lines", cv::WINDOW_NORMAL);
			cv::imshow("lines", circles);
		}*/

		/*// 13. Harris-Corner-Detection
		{
			cv::Mat grey;
			cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

			std::vector<cv::KeyPoint> key_points;
			float alpha = 0.04f;

			int th = 50000;

			myHarrisCornerDetector(grey, key_points, th, alpha);
			cv::Mat k_points;
			cv::drawKeypoints(grey, key_points, k_points, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

			// display image
			cv::namedWindow("k_points", cv::WINDOW_NORMAL);
			cv::imshow("k_points", k_points);
		}*/

		////////////////////////

		// wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout << "key " << int(key) << std::endl;

		// here you can implement some looping logic using key value:
		//  - pause
		//  - stop_left_x
		//  - step back
		//  - step forward
		//  - loop on the same frame

		switch (key)
		{
		case 'p':
			std::cout << "Mat = " << std::endl
					  << image << std::endl;
			break;
		case 'q':
			exit_loop = 1;
			break;
		case 'c':
			std::cout << "SET COLOR imread()" << std::endl;
			imreadflags = cv::IMREAD_COLOR;
			break;
		case 'g':
			std::cout << "SET GREY  imread()" << std::endl;
			imreadflags = cv::IMREAD_GRAYSCALE; // Y = 0.299 R + 0.587 G + 0.114 B
			break;
		}

		frame_number++;
	}

	return 0;
}

#if 0
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  args.wait_t=0;

  cv::CommandLineParser parser(argc, argv,
      "{input   i|in.png|input image, Use %0xd format for multiple images.}"
      "{wait    t|0     |wait before next frame (ms)}"
      "{help    h|<none>|produce help message}"
      );

  if(parser.has("help"))
  {
    parser.printMessage();
    return false;
  }

  args.image_name = parser.get<std::string>("input");
  args.wait_t     = parser.get<int>("wait");

  return true;
}
#else

#include <unistd.h>
bool ParseInputs(ArgumentList &args, int argc, char **argv)
{
	int c;
	args.threshold = 40;
	args.k = 5;
	args.alpha = 0.5f;
	args.minThreshold = 50;

	while ((c = getopt(argc, argv, "hi:t:c:p:")) != -1)
		switch (c)
		{
		case 't':
			args.wait_t = atoi(optarg);
			break;
		case 'i':
			args.image_name = optarg;
			break;
		case 'c':
			args.top_left_x = atoi(optarg);
			args.top_left_y = atoi(optarg);
			args.w = atoi(optarg);
			args.h = atoi(optarg);
			break;
		case 'p':
			args.padding_size = atoi(optarg);
		case 'h':
		default:
			std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
			std::cout << "exit:  type q" << std::endl
					  << std::endl;
			std::cout << "Allowed options:" << std::endl
					  << "   -h                       produce help message" << std::endl
					  << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
					  << "   -t arg                   wait before next frame (ms)" << std::endl
					  << "   -c arg                   crop image. Use top_left_x top_left_y w h" << std::endl
					  << "   -p arg                   pad image. Use padding_size"
					  << std::endl;
			return false;
		}
	return true;
}

#endif

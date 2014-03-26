#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp" 
#include "Image.h"

// We'll throw out the GPU analysis for now. What we do know that openCV is fully capable of supporting NVIDIA's CUDA
// Will look into this later...

using namespace std;
using namespace cv; 
using namespace flow_CLIU;

// Globals for later usage
int finalFlowNum;
int frameNum;
int frameWidth;
int frameHeight;
// The retvalued file
float alpha;
float advectcap;

// Converts Matrix frame to DImage
DImage* convertFrameToDImage(Mat* frame)
{
	DImage* output = new DImage(frameWidth, frameHeight, 3);

	for (int y = 0; y < frameHeight; y++)
	{
		for (int x = 0; x < frameWidth; x++)
		{
			output->pData[3 * (frameWidth * y + x)] = ((double) frame->at<Vec3b>(y, x)[0])/255.0;
			output->pData[3 * (frameWidth * y + x) + 1] = ((double) frame->at<Vec3b>(y, x)[1])/255.0;
			output->pData[3 * (frameWidth * y + x) + 2] = ((double) frame->at<Vec3b>(y, x)[2])/255.0;
		}
	}

	// Return the finalized frame;
	return output;
}

// Initializes vector bitmap for frame to double image format
DImage* convertFieldToDImage(FILE *field)
{
	DImage* output = new DImage(frameWidth, frameHeight, 2);
	float temp;
	float negAdvect = -1 * advectcap;

	// Converts the field into DImage format
	for (int y = 0; y < frameHeight; y++)
	{
		for (int x = 0; x < frameWidth; x++)
		{
			for (int i = 0; i < 2; i++)
			{
				fread_s(&temp, sizeof(float), sizeof(float), 1, field);
				output->pData[2 * (frameWidth * y + x) + i] = (double) max(negAdvect, min(advectcap, alpha * temp));
			}
		}
	}

	// Return the finalized frame
	return output;
}

// Clamps a color value to within proper boundaries
unsigned char clampColor(double origColor)
{
	if (origColor < 0)
		return 0;
	if (origColor > 1)
		return 255;
	else
		return (unsigned char) 255 * origColor;
}

// Runs backward interative interp on initialized image field
Mat* backIterativeInterp(DImage *prevField)
{
	DImage dx, dy, dxdy;
	prevField->computeGradient(dx, dy, dxdy);

	// Check to make sure this is cast right
	Mat *output = new Mat(frameWidth, frameHeight, CV_64FC2);

	double *newVector = new double[2];
	double dVector[2];

	// Start mapping p properly
	for (int y = 0; y < frameHeight; y++)
	{
		for (int x = 0; x < frameWidth; x++)
		{
			// Check this later
			output->at<Vec2d>(y, x)[0] = x;
			output->at<Vec2d>(y, x)[1] = y;

			for (int i = 0; i < 5; i++)
			{
				// This might cause casting errors later
				dVector[0] = output->at<Vec2d>(y, x)[0];
				dVector[1] = output->at<Vec2d>(y, x)[1];

				// If bicube falls within frame we can backstep
				if (prevField->bicubicInterp(&newVector, dVector, dx, dy, dxdy))
				{
					// Check to make sure this is right (ESPECIALLY!!!)
					output->at<Vec2d>(y, x)[0] = x - newVector[0];
					output->at<Vec2d>(y, x)[1] = y - newVector[1];
				}
				// If it falls outside the frame we can no longer backstep
				else 
				{
					break;
				}
			}
		}
	}

	// Return the result
	return output;
}

// Correct newFrameField later
Mat* magnifyVideoInward(Mat* frame, FILE *fp)
{
	DImage dx, dy, dxdy;
	DImage *dframe = convertFrameToDImage(frame);
	Mat *dfield = backIterativeInterp(convertFieldToDImage(fp));

	Mat *output = new Mat(frameWidth, frameHeight, CV_8UC3);

	dframe->computeGradient(dx, dy, dxdy);

	// Var for storing process
	double *newPixel = new double[3];
	double dVector[2];

	// Looping through and adjusting each pixel
	for (int y = 0; y < frameHeight; y++)
	{
		for (int x = 0; x < frameWidth; x++)
		{
			// This might cause casting errors later
			dVector[0] = dfield->at<Vec2d>(y, x)[0];
			dVector[1] = dfield->at<Vec2d>(y, x)[1];

			// Runs the bicubic interp
			// If success use this new pixel
			if (dframe->bicubicInterp(&newPixel, dVector, dx, dy, dxdy))
			{
				output->at<Vec3b>(y, x)[0] = clampColor(newPixel[0]);
				output->at<Vec3b>(y, x)[1] = clampColor(newPixel[1]);
				output->at<Vec3b>(y, x)[2] = clampColor(newPixel[2]);
			}
			// Else use generic grey pixel
			else
			{
				output->at<Vec3b>(y, x)[0] = 128;
				output->at<Vec3b>(y, x)[1] = 128;
				output->at<Vec3b>(y, x)[2] = 128;
			}
		}
	}

	// Return finalized frame (output)
	return output;
}

// The main function
int main (int argc, char *argv[])
{
	// Some init variables... will be used later
	alpha = 4;
	advectcap = 10;

	const char *inputVideoPath = "C:/Users/Allen Qiu/Documents/Visual Studio 2012/Projects/openCVtest246/guitar.mp4";
	const char *inputBitmapPath = "C:/Users/Allen Qiu/Documents/Visual Studio 2012/Projects/openCVtest246/guitar_wfield.bin";
	const char *outputVideoPath = "C:/Users/Allen Qiu/Documents/Visual Studio 2012/Projects/openCVtest246/guitar_new.mp4";

	// Gets the video from file
	VideoCapture inputVideo(inputVideoPath);
	VideoWriter outputVideo;
	
	// Gets the bitmap from file
	FILE *fp = NULL;
	fopen_s(&fp, inputBitmapPath, "r");	// Apparently this is safer...

	// In case capturing video fails
	if (!inputVideo.isOpened())
	{
		cout << "Error: video capture failed" << endl;
		return -1;
	}

	// In case file retrieval fails
	if (fp == NULL)
	{
		cout << "Error: bitmap retrieval failed" << endl;
		return -1;
	}

	// Else both worked -> init the variables
	fread(&frameNum, sizeof(int), 1, fp);
	fread(&frameHeight, sizeof(int), 1, fp);
	fread(&frameWidth, sizeof(int), 1, fp);

	int askFileTypeBox = 0;
	bool color = true;
	double frameRate = 33;
	Size size = Size(frameHeight, frameWidth);

	outputVideo.open(outputVideoPath, askFileTypeBox, frameRate, size, color);

	// The fp file should now be pointing to start of bitmap
	// Look at this later IN CASE WE WANNA CHANGE THE FLOW NUMBER
	// Need some kind of skip function q

	namedWindow("Finished Display", 1);

	Mat frame;
	inputVideo >> frame;

	while (true)
	{
		inputVideo >> frame;
		if (frame.empty())
			break;

		Mat * newFrame = magnifyVideoInward(&frame, fp);

		imshow("Finished Display", *newFrame);
		outputVideo << frame;
	}

	cout << "Finished processing video" << endl;
}
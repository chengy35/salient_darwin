#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_

#include "DenseTrackStab.h"
using namespace cv;

// get the rectangle for computing the descriptor
void GetRect(const Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo)
{
	int x_min = descInfo.width/2;
	int y_min = descInfo.height/2;
	int x_max = width - descInfo.width;
	int y_max = height - descInfo.height;

	rect.x = std::min<int>(std::max<int>(cvRound(point.x) - x_min, 0), x_max);
	rect.y = std::min<int>(std::max<int>(cvRound(point.y) - y_min, 0), y_max);
	rect.width = descInfo.width;
	rect.height = descInfo.height;
}

// compute integral histograms for the whole image
void BuildDescMat(const Mat& xComp, const Mat& yComp, float* desc, const DescInfo& descInfo, CvMat* kernelMatrix) // kernel matrix of densely sampled angles
{
	float maxAngle = 360.f;
	int nBins = descInfo.isHof ? descInfo.nBins-1 : descInfo.nBins;
	float denseBase = (2 * M_PI) / float(kernelMatrix->height);
	
	int width = xComp.cols;
	int height = xComp.rows;
	int histDim = descInfo.nBins;
	int index = 0;

	for(int i = 0; i < height; i++)
	{

		const float* xc = xComp.ptr<float>(i);
		const float* yc = yComp.ptr<float>(i);

		// summarization of the current line
		std::vector<float> sum(histDim);
		for(int j = 0; j < xComp.cols; j++,index++)
		{
			float shiftX = xc[j];
			float shiftY = yc[j];
			float magnitude = sqrt(shiftX*shiftX+shiftY*shiftY);
			
			// for the zero bin of hof
			if(descInfo.isHof && magnitude <= min_flow) 
			{
				int bin = nBins; // the zero bin is the last one
				magnitude = 1;
				sum[bin] += magnitude;
			}
			else
			{
				float orientation = atan2(shiftY, shiftX);
				if (orientation < 0)
				{
					orientation += 2 * M_PI;
				}
				
				int iDense = static_cast<int>(roundf(orientation / denseBase));
				if (iDense >= kernelMatrix->height)
				{
					iDense = 0;
				}
				// directly apply kernel histograms
				float* ptr = (float*)(kernelMatrix->data.ptr + iDense * kernelMatrix->step);
				for (int m = 0; m < nBins; m++)
				{
					sum[m] += magnitude * ptr[m];
				}
			}

			int temp0 = index*nBins;
			if(i == 0)
			{ // for the first line
				for(int m = 0; m <nBins; m++)
					desc[temp0++] = sum[m];
			}
			else
			{
				int temp1 = (index - width)*nBins;
				for(int m = 0; m <nBins; m++)
					desc[temp0++] = desc[temp1++]+sum[m];
			}
		}
	}
}

// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index)
{
	int dim = descInfo.dim;
	int nBins = descInfo.nBins;
	int height = descMat->height;
	int width = descMat->width;

	int xStride = rect.width/descInfo.nxCells;
	int yStride = rect.height/descInfo.nyCells;
	int xStep = xStride*nBins;
	int yStep = yStride*width*nBins;

	// iterate over different cells
	int iDesc = 0;
	std::vector<float> vec(dim);
	for(int xPos = rect.x, x = 0; x < descInfo.nxCells; xPos += xStride, x++)
	for(int yPos = rect.y, y = 0; y < descInfo.nyCells; yPos += yStride, y++) {
		// get the positions in the integral histogram
		const float* top_left = descMat->desc + (yPos*width + xPos)*nBins;
		const float* top_right = top_left + xStep;
		const float* bottom_left = top_left + yStep;
		const float* bottom_right = bottom_left + xStep;

		for(int i = 0; i < nBins; i++) {
			float sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
			vec[iDesc++] = std::max<float>(sum, 0) + epsilon;
		}
	}

	float norm = 0;
	for(int i = 0; i < dim; i++)
		norm += vec[i];
	if(norm > 0) norm = 1./norm;

	int pos = index*dim;
	for(int i = 0; i < dim; i++)
		desc[pos++] = sqrt(vec[i]*norm);
}

// for HOG descriptor
void HogComp(const Mat& img, float* desc, DescInfo& descInfo, CvMat* kernelMatrix)
{
	Mat imgX, imgY;
	Sobel(img, imgX, CV_32F, 1, 0, 1);
	Sobel(img, imgY, CV_32F, 0, 1, 1);
	BuildDescMat(imgX, imgY, desc, descInfo,kernelMatrix);
}

// for HOF descriptor
void HofComp(const Mat& flow, float* desc, DescInfo& descInfo, CvMat* kernelMatrix)
{
	Mat flows[2];
	split(flow, flows);
	BuildDescMat(flows[0], flows[1], desc, descInfo,kernelMatrix);
}

// for MBH descriptor
void MbhComp(const Mat& flow, float* descX, float* descY, DescInfo& descInfo, CvMat* kernelMatrix)
{
	Mat flows[2];
	split(flow, flows);

	Mat flowXdX, flowXdY, flowYdX, flowYdY;
	Sobel(flows[0], flowXdX, CV_32F, 1, 0, 1);
	Sobel(flows[0], flowXdY, CV_32F, 0, 1, 1);
	Sobel(flows[1], flowYdX, CV_32F, 1, 0, 1);
	Sobel(flows[1], flowYdY, CV_32F, 0, 1, 1);

	BuildDescMat(flowXdX, flowXdY, descX, descInfo,kernelMatrix);
	BuildDescMat(flowYdX, flowYdY, descY, descInfo,kernelMatrix);
}

// check whether a trajectory is valid or not
bool IsValid(std::vector<Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
	int size = track.size();
	float norm = 1./size;
	for(int i = 0; i < size; i++) {
		mean_x += track[i].x;
		mean_y += track[i].y;
	}
	mean_x *= norm;
	mean_y *= norm;

	for(int i = 0; i < size; i++) {
		float temp_x = track[i].x - mean_x;
		float temp_y = track[i].y - mean_y;
		var_x += temp_x*temp_x;
		var_y += temp_y*temp_y;
	}
	var_x *= norm;
	var_y *= norm;
	var_x = sqrt(var_x);
	var_y = sqrt(var_y);

	// remove static trajectory
	if(var_x < min_var && var_y < min_var)
		return false;
	// remove random trajectory
	if( var_x > max_var || var_y > max_var )
		return false;

	float cur_max = 0;
	for(int i = 0; i < size-1; i++) {
		track[i] = track[i+1] - track[i];
		float temp = sqrt(track[i].x*track[i].x + track[i].y*track[i].y);

		length += temp;
		if(temp > cur_max)
			cur_max = temp;
	}

	if(cur_max > max_dis && cur_max > length*0.7)
		return false;

	track.pop_back();
	norm = 1./length;
	// normalize the trajectory
	for(int i = 0; i < size-1; i++)
		track[i] *= norm;

	return true;
}

bool IsCameraMotion(std::vector<Point2f>& disp)
{
	float disp_max = 0;
	float disp_sum = 0;
	for(int i = 0; i < disp.size(); ++i) {
		float x = disp[i].x;
		float y = disp[i].y;
		float temp = sqrt(x*x + y*y);

		disp_sum += temp;
		if(disp_max < temp)
			disp_max = temp;
	}

	if(disp_max <= 1)
		return false;

	float disp_norm = 1./disp_sum;
	for (int i = 0; i < disp.size(); ++i)
		disp[i] *= disp_norm;

	return true;
}

// detect new feature points in an image without overlapping to previous points
void DenseSample(const Mat& grey, std::vector<Point2f>& points, const double quality, const int min_distance)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);

	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters(width*height);
	int x_max = min_distance*width;
	int y_max = min_distance*height;

	for(int i = 0; i < points.size(); i++) {
		Point2f point = points[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);

		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters[y*width+x]++;
	}

	points.clear();
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters[index] > 0)
			continue;

		int x = j*min_distance+offset;
		int y = i*min_distance+offset;

		if(eig.at<float>(y, x) > threshold)
			points.push_back(Point2f(float(x), float(y)));
	}
}

void InitPry(const Mat& frame, std::vector<float>& scales, std::vector<Size>& sizes)
{
	int rows = frame.rows, cols = frame.cols;
	float min_size = std::min<int>(rows, cols);

	int nlayers = 0;
	while(min_size >= patch_size) {
		min_size /= scale_stride;
		nlayers++;
	}

	if(nlayers == 0) nlayers = 1; // at least 1 scale 

	scale_num = std::min<int>(scale_num, nlayers);

	scales.resize(scale_num);
	sizes.resize(scale_num);

	scales[0] = 1.;
	sizes[0] = Size(cols, rows);

	for(int i = 1; i < scale_num; i++) {
		scales[i] = scales[i-1] * scale_stride;
		sizes[i] = Size(cvRound(cols/scales[i]), cvRound(rows/scales[i]));
	}
}

void BuildPry(const std::vector<Size>& sizes, const int type, std::vector<Mat>& grey_pyr)
{
	int nlayers = sizes.size();
	grey_pyr.resize(nlayers);

	for(int i = 0; i < nlayers; i++)
		grey_pyr[i].create(sizes[i], type);
}

void DrawTrack(const std::vector<Point2f>& point, const int index, const float scale, Mat& image)
{
	Point2f point0 = point[0];
	point0 *= scale;

	for (int j = 1; j <= index; j++) {
		Point2f point1 = point[j];
		point1 *= scale;
		line(image, point0, point1, Scalar(0,cvFloor(255.0*(j+1.0)/index),0), 2, 8, 0);
		point0 = point1;
	}
	circle(image, point0, 2, Scalar(255,0,0), -1, 8, 0);
}

void PrintDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo)
{
	int tStride = cvFloor(trackInfo.length/descInfo.ntCells);
	float norm = 1./float(tStride);
	int dim = descInfo.dim;
	int pos = 0;
	for(int i = 0; i < descInfo.ntCells; i++) {
		std::vector<float> vec(dim);
		for(int t = 0; t < tStride; t++)
			for(int j = 0; j < dim; j++)
				vec[j] += desc[pos++];
		for(int j = 0; j < dim; j++)
			printf("%.7f\t", vec[j]*norm);
	}
}

void LoadBoundBox(char* file, std::vector<Frame>& bb_list)
{
	// load the bouding box file
    std::ifstream bbFile(file);
    std::string line;

    while(std::getline(bbFile, line)) {
		 std::istringstream iss(line);

		int frameID;
		if (!(iss >> frameID))
			continue;

		Frame cur_frame(frameID);

		float temp;
		std::vector<float> a(0);
		while(iss >> temp)
			a.push_back(temp);

		int size = a.size();

		if(size % 5 != 0)
			fprintf(stderr, "Input bounding box format wrong!\n");

		for(int i = 0; i < size/5; i++)
			cur_frame.BBs.push_back(BoundBox(a[i*5], a[i*5+1], a[i*5+2], a[i*5+3], a[i*5+4]));

		bb_list.push_back(cur_frame);
    }
}

void InitMaskWithBox(Mat& mask, std::vector<BoundBox>& bbs)
{
	int width = mask.cols;
	int height = mask.rows;

	for(int i = 0; i < height; i++) {
		uchar* m = mask.ptr<uchar>(i);
		for(int j = 0; j < width; j++)
			m[j] = 1;
	}

	for(int k = 0; k < bbs.size(); k++) {
		BoundBox& bb = bbs[k];
		for(int i = cvCeil(bb.TopLeft.y); i <= cvFloor(bb.BottomRight.y); i++) {
			uchar* m = mask.ptr<uchar>(i);
			for(int j = cvCeil(bb.TopLeft.x); j <= cvFloor(bb.BottomRight.x); j++)
				m[j] = 0;
		}
	}
}

static void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags = INTER_LINEAR,
	            			 int borderType = BORDER_CONSTANT, const Scalar& borderValue = Scalar())
{
	int width = src.cols;
	int height = src.rows;
	dst.create( height, width, CV_8UC1 );

	Mat mask = Mat::zeros(height, width, CV_8UC1);
	const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
         invert(matM, matM);

    int x, y, x1, y1;

    int bh0 = std::min(BLOCK_SZ/2, height);
    int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
    bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

    for( y = 0; y < height; y += bh0 ) {
    for( x = 0; x < width; x += bw0 ) {
		int bw = std::min( bw0, width - x);
        int bh = std::min( bh0, height - y);

        Mat _XY(bh, bw, CV_16SC2, XY);
		Mat matA;
        Mat dpart(dst, Rect(x, y, bw, bh));

		for( y1 = 0; y1 < bh; y1++ ) {

			short* xy = XY + y1*bw*2;
            double X0 = M[0]*x + M[1]*(y + y1) + M[2];
            double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
            double W0 = M[6]*x + M[7]*(y + y1) + M[8];
            short* alpha = A + y1*bw;

            for( x1 = 0; x1 < bw; x1++ ) {

                double W = W0 + M[6]*x1;
                W = W ? INTER_TAB_SIZE/W : 0;
                double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
                double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
 
				double _X = fX/double(INTER_TAB_SIZE);
				double _Y = fY/double(INTER_TAB_SIZE);

				if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
					mask.at<uchar>(y+y1, x+x1) = 1;

                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);

                xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
            }
        }

        Mat _matA(bh, bw, CV_16U, A);
        remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
    }
    }

	for( y = 0; y < height; y++ ) {
		const uchar* m = mask.ptr<uchar>(y);
		const uchar* s = prev_src.ptr<uchar>(y);
		uchar* d = dst.ptr<uchar>(y);
		for( x = 0; x < width; x++ ) {
			if(m[x] == 0)
				d[x] = s[x];
		}
	}
}

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
				  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
	prev_pts.clear();
	pts.clear();

	if(prev_kpts.size() == 0 || kpts.size() == 0)
		return;

	Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

	BFMatcher desc_matcher(NORM_L2);
	std::vector<DMatch> matches;

	desc_matcher.match(desc, prev_desc, matches, mask);
	
	prev_pts.reserve(matches.size());
	pts.reserve(matches.size());

	for(size_t i = 0; i < matches.size(); i++) {
		const DMatch& dmatch = matches[i];
		// get the point pairs that are successfully matched
		prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
		pts.push_back(kpts[dmatch.queryIdx].pt);
	}

	return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
				const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
				std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
	prev_pts_all.clear();
	prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

	pts_all.clear();
	pts_all.reserve(pts1.size() + pts2.size());

	for(size_t i = 0; i < prev_pts1.size(); i++) {
		prev_pts_all.push_back(prev_pts1[i]);
		pts_all.push_back(pts1[i]);
	}

	for(size_t i = 0; i < prev_pts2.size(); i++) {
		prev_pts_all.push_back(prev_pts2[i]);
		pts_all.push_back(pts2[i]);	
	}

	return;
}

void MatchFromFlow(const Mat& prev_grey, const Mat& flow, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);
	
	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f = flow.ptr<float>(y);
		pts.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
	}
}



//--------------------------------begin--------------------------------
//-------------------------modified by Yikun Lin-----------------------
/* tracking interest points by median filtering in the optical field */
void OpticalFlowTracker(Mat& flow_mat, // the optical field
						CvMat* salMap, // saliency map
						std::vector<CvPoint2D32f>& points_in, // input interest point positions
						std::vector<CvPoint2D32f>& points_out, // output interest point positions
						std::vector<int>& status, // status for successfully tracked or not
						std::vector<float>& saliency) // the saliency value to output
{
	IplImage flow(flow_mat);  //(1)  
	if(points_in.size() != points_out.size())
		fprintf(stderr, "the numbers of points don't match!");
	if(points_in.size() != status.size())
		fprintf(stderr, "the number of status doesn't match!");
	int width = flow.width;
	int height = flow.height;

	for(int i = 0; i < points_in.size(); i++) {
		CvPoint2D32f point_in = points_in[i];
		std::list<float> xs;
		std::list<float> ys;
		std::list<float> ss;
		int x = cvFloor(point_in.x);
		int y = cvFloor(point_in.y);
		for(int m = x-1; m <= x+1; m++)
		for(int n = y-1; n <= y+1; n++) {
			int p = std::min<int>(std::max<int>(m, 0), width-1);
			int q = std::min<int>(std::max<int>(n, 0), height-1);
			const float* f = (const float*)(flow.imageData + flow.widthStep*q);
			xs.push_back(f[2*p]);
			ys.push_back(f[2*p+1]);
			f = (const float*)(salMap->data.ptr + q * salMap->step);
			ss.push_back(f[p]);
		}

		xs.sort();
		ys.sort();
		ss.sort();
		int size = xs.size()/2;
		for(int m = 0; m < size; m++) {
			xs.pop_back();
			ys.pop_back();
			ss.pop_back();
		}

		CvPoint2D32f offset;
		offset.x = xs.back();
		offset.y = ys.back();
		saliency[i] = ss.back();
		CvPoint2D32f point_out;
		point_out.x = point_in.x + offset.x;
		point_out.y = point_in.y + offset.y;
		points_out[i] = point_out;
		
		if( point_out.x > 0 && point_out.x < width && point_out.y > 0 && point_out.y < height )
			status[i] = 1;
		else
			status[i] = -1;
	}
}

float getSum(Mat mat_CV_8UC1 )
{
   float averageSaliency = 0;
  
   for( size_t nrow = 0; nrow < mat_CV_8UC1.rows; nrow++)  
   {
   	   uchar* data = mat_CV_8UC1.ptr<uchar>(nrow);
       for(size_t ncol = 0; ncol < mat_CV_8UC1.cols; ncol++)  
       {  
          averageSaliency += (float(data[ncol]));
          //cout<<(int)data[ncol]<<endl;
       }
   }
   averageSaliency /= ( mat_CV_8UC1.rows* mat_CV_8UC1.cols);
   return averageSaliency;
}

/* check whether a trajectory is salient or not */
int isValid(std::vector<Point2f>& track, std::vector<float>& saliency, std::vector<float>& averageSaliency, float threshold)
{
	/*int size = track.size();
	float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
	for(int i = 0; i < size; i++) {
		mean_x += track[i].x;
		mean_y += track[i].y;
	}
	mean_x /= size;
	mean_y /= size;

	for(int i = 0; i < size; i++) {
		track[i].x -= mean_x;
		var_x += track[i].x*track[i].x;
		track[i].y -= mean_y;
		var_y += track[i].y*track[i].y;
	}
	var_x /= size;
	var_y /= size;
	var_x = sqrt(var_x);
	var_y = sqrt(var_y);
	// remove static trajectory
	if(var_x < min_var && var_y < min_var)
		return 0;
	// remove random trajectory
	if( var_x > max_var || var_y > max_var )
		return 0;

	for(int i = 1; i < size; i++) {
		float temp_x = track[i].x - track[i-1].x;
		float temp_y = track[i].y - track[i-1].y;
		length += sqrt(temp_x*temp_x+temp_y*temp_y);
		track[i-1].x = temp_x;
		track[i-1].y = temp_y;
	}

	float len_thre = length*0.7;
	for( int i = 0; i < size-1; i++ ) {
		float temp_x = track[i].x;
		float temp_y = track[i].y;
		float temp_dis = sqrt(temp_x*temp_x + temp_y*temp_y);
		if( temp_dis > max_dis && temp_dis > len_thre )
			return 0;
	}

	track.pop_back();
	// normalize the trajectory
	for(int i = 0; i < size-1; i++) {
		track[i].x /= length;
		track[i].y /= length;
	}
	return 1;*/
	int size = track.size();
	// keep salient trajectories
	float averageSal = 0;
	float sal = 0;
	for (int i = 1; i < size; i++)
	{
		averageSal += averageSaliency[i];
		sal += saliency[i];
	}
	averageSal /= (size - 1);
	sal /= (size - 1);
	if (sal >= averageSal * threshold)
	{
		return 1;
	}
	return 0;
}
//---------------------------------end---------------------------


#endif /*DESCRIPTORS_H_*/

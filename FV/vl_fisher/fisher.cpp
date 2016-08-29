#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <zlib.h>
#include <vector>
#include <string.h>
#include <set>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "generial.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
extern "C"
{
	#include <vl/generic.h>
	#include <vl/fisher.h>
}

using namespace std;
using namespace cv;
void saveDatatoFile(float* trjData,int size,char *filepath)
{
	ofstream file(filepath);
	cout<<size<<" is the size"<<endl;
	for (size_t  i = 0;  i < size; i ++) {
		file << trjData[i]<<" ";
	}
	file.close();
	cout<<filepath<<" saved"<<endl;
}
void saveVectorFile(std::vector<vector<float> > *trjData,char *filepath)
{
	ofstream file(filepath);
	for (size_t  i = 0;  i < (*trjData).size(); i ++) {
		for (int j = 0; j < TRJ_DI; ++j)
		{
			file<<(*trjData)[i][j]<<" ";
		}
		file<<endl;
	}
	file.close();
	cout<<filepath<<" saved"<<endl;
}
char * ReadLine(gzFile gzfp)
{
	int len;
	if(gzgets(gzfp, iiline, max_line_len) == NULL)
		return NULL;
	while(strrchr(iiline, '\n') == NULL)
	{
		max_line_len *= 2;
		iiline = (char*) realloc(iiline, max_line_len);
		len = (int) strlen(iiline);
		if(gzgets(gzfp, iiline + len, max_line_len - len) == NULL)
			break;
	}
	return iiline;
}
void saveFisherVectorToFile(float * fv_trj, int dimension,char *fv_file)
{
	ofstream file(fv_file,ios::app);
	
	for (size_t s = 0; s < dimension; s++)
	{
		file << fv_trj[s]<<" ";
	}
	file<<endl;

	file.close();
	//cout<<fv_file<<" saved"<<endl;
}
void readGmmFromFile(float *data, char *fileName)
{
	cout<<"read Gmm from  "<<fileName<<endl;
	ifstream _file(fileName);
	if(_file)
	{
		int index = 0;
		while(!_file.eof())
		{
			_file >> data[index++];
		}
		_file.close();
	}
	else{
		cout<<fileName<<" do not exist"<<endl;
		return ;
	}
}
//void getDescriptorFromFile(char *descriptorFileName,vector<int>*obj,vector<vector<float> > *trj,vector<vector<float> > *hog,vector<vector<float> > *hof,vector<vector<float> > *mbh)
void getDescriptorFromFile(char *descriptorFileName,vector<int>*obj,vector<vector<float> > *mbh)
{
	printf("%s\n", "get DescriptorFromFile function to get fisher vector");
	int nLines = 0;
	max_line_len = 1024;
	iiline = Malloc(char, max_line_len);
	gzFile file = gzopen(descriptorFileName,"r");
	float value;
	int value2;
	vector<float> temp;

	while (ReadLine(file) != NULL)
	{
		//printf("%s\n","readline" );
		int j;
		char* feature;
		feature = strtok(iiline," \t");
		value2 = atoi(feature);
		obj->push_back(value2);

		for (j = 0; j < OBJ_DI-1; j++)
		{
			feature = strtok(NULL," \t"); // get rid of obj and trj
		}
		temp.clear();
		for (j = 0; j< TRJ_DI; j++)
		{
			feature = strtok(NULL," \t");
			//value = atof(feature);
			//temp.push_back(value);
		}
		//trj->push_back(temp);
		//temp.clear();
		for (j = 0; j< HOG_DI; j++)
		{
			feature = strtok(NULL," \t");
			// value = atof(feature);
			// value = sqrt(value);
			// temp.push_back(value);
		}
		// hog->push_back(temp);
		// temp.clear();
		for (j = 0; j < HOF_DI; j++)
		{
			feature = strtok(NULL," \t");
			// value = atof(feature);
			// value = sqrt(value);
			// temp.push_back(value);

		}
		// hof->push_back(temp);
		temp.clear();
		for (j = 0; j < MBH_DI; j++)
		{
			feature = strtok(NULL," \t");
			value = atof(feature);
			value = sqrt(value);
			temp.push_back(value);
		}
		mbh->push_back(temp);
	}
	temp.clear();
	free(iiline);
	//printf("%s, and frame-size is ,%d \n","the end of features",(*obj).size());
	//printf("%s, %d and feature size is ,%d \n","the end of features",(*mbh).size(),(*mbh)[0].size() );
	gzclose(file);
}


float* computPCAandReduce(vector<vector<float> > *data,int startIndex, int DIMENTIONS ,int SAMPLE_NUM, double pcaFactor)
{
	//cout<<startIndex<<" is start"<<endIndex<<" is end "<< DIMENTIONS<<" is dimension"<<SAMPLE_NUM<<" is sample number"<<pcaFactor<<" is factor"<<endl;
	Mat input(SAMPLE_NUM, DIMENTIONS, CV_32FC1); //原始数据
    for (size_t i = 0; i < SAMPLE_NUM; i++)
    {
        for (int j = 0; j < DIMENTIONS;j++)
        {
            input.at<float>(i, j) = (*data)[i+startIndex][j];
        }
    }
    PCA *pca_encoding  = new PCA();
    char *filepath = new char [100];
	if(DIMENTIONS == TRJ_DI)
		strcpy(filepath,pcaTrjInfo);
	else if(DIMENTIONS == HOG_DI)
		strcpy(filepath,pcaHogInfo);
	else if(DIMENTIONS == HOF_DI)
		strcpy(filepath,pcaHofInfo);
	else if(DIMENTIONS == MBH_DI)
		strcpy(filepath,pcaMbhInfo);

	FileStorage fs_r(filepath, FileStorage::READ);
	fs_r[PCA_MEAN] >> pca_encoding->mean;
	fs_r[PCA_EIGEN_VECTOR] >> pca_encoding->eigenvectors;
	fs_r.release();

	Mat output_encode(SAMPLE_NUM, pca_encoding->eigenvectors.rows, CV_32FC1);
	pca_encoding->project(input, output_encode);

	float * oneDimResult = new float[int(DIMENTIONS*pcaFactor*SAMPLE_NUM)+1];
	int oneIndex = 0;
	for (int i = 0; i < SAMPLE_NUM;i++) // the reduced dimension  matrix. (pca)
    {
        for (int j = 0; j < DIMENTIONS*pcaFactor;j++)
        {
            oneDimResult[oneIndex++] = output_encode.at<float>(i,j);// index is 0, for interface gmm...getAndSaveGmmModel
        }
    }
    input.release();
    output_encode.release();
	delete pca_encoding;
	delete filepath;
	return oneDimResult;
}

void saveObjDatatoFile(std::vector<int> obj, char *fileName)
{
	ofstream file(fileName);
	//cout<<obj.size()<<"===================is obj's size"<<endl;
	for (int i = 0; i < obj.size(); ++i)
	{
		file<<obj[i]<<endl;
	}
	file.close();
	return ;
}


void getAndSaveFV (char *descriptorFileName, int gmmSize, float * Mbhmeans,float *  Mbhcovariances,float *  Mbhpriors, 
	char * feat_mbh_fv_file)
{
	vector<int> obj;
	vector<vector<float> > mbh;
	obj.clear();
	mbh.clear();
	getDescriptorFromFile(descriptorFileName,&obj,&mbh);

	cout<<"after select descriptor from file "<<endl;
	int allTrjSize = obj.size();
	cout<<allTrjSize<<" is the trj size"<<endl;
	
	cout<<"before pca"<<endl;

	PCA *pca_encoding  = new PCA();
    char *filepath = new char [100];
	strcpy(filepath,pcaMbhInfo);
	cout<<"before read pca from file "<<endl;
	FileStorage fs_r(filepath, FileStorage::READ);
	fs_r[PCA_MEAN] >> pca_encoding->mean;
	fs_r[PCA_EIGEN_VECTOR] >> pca_encoding->eigenvectors;
	fs_r.release();
	cout<<"after read pca from file "<<endl;
	Mat input(1, MBH_DI, CV_32FC1); //原始数据
	Mat output_encode(1, pca_encoding->eigenvectors.rows, CV_32FC1);
	
	int indexLineofFisher = 0;
	int numDataToEncode = 1;
	
	float *fv_mbh = new float[(int)(2*pcaFactor*MBH_DI*gmmSize)];

	for (vector<int>::iterator it = obj.begin();it != obj.end(); it++)
	{
		void * encTrj = vl_malloc(sizeof(float) * 2 * pcaFactor* MBH_DI * gmmSize);
		float * reducedTRJData = computPCAandReduce(&mbh,indexLineofFisher,MBH_DI,numDataToEncode,pcaFactor);
	
	    for (size_t i = 0; i < 1; i++)
	    {
	        for (int j = 0; j < MBH_DI;j++)
	        {
	            input.at<float>(i, j) = mbh[i+indexLineofFisher][j];
	        }
	    }
	    //cout<<"after project "<<endl;
		pca_encoding->project(input, output_encode);

		int oneIndex = 0;
		for (int i = 0; i < 1;i++) // the reduced dimension  matrix. (pca)
	    {
	        for (int j = 0; j < MBH_DI*pcaFactor;j++)
	        {
	            reducedTRJData[oneIndex++] = output_encode.at<float>(i,j);// index is 0, for interface gmm...getAndSaveGmmModel
	        }
	    }

		vl_fisher_encode(encTrj,VL_TYPE_FLOAT,Mbhmeans,pcaFactor*MBH_DI, gmmSize, 
		Mbhcovariances,Mbhpriors,(void *)reducedTRJData, numDataToEncode,VL_FISHER_FLAG_IMPROVED);
		for (size_t i = 0; i < 2*pcaFactor*MBH_DI*gmmSize; i++)
		{
			fv_mbh[i] = ((float * )encTrj)[i];
		}
		vl_free(encTrj);

		saveFisherVectorToFile(fv_mbh,(int)(2*pcaFactor*MBH_DI*gmmSize),feat_mbh_fv_file);
		
		indexLineofFisher++;
		delete reducedTRJData;
	}
	cout<<feat_mbh_fv_file<<" file saved !"<<endl;
	input.release();
    output_encode.release();
	delete pca_encoding;
	delete filepath;
	delete []fv_mbh;
}


int main(int argc, char const *argv[]) {

	char ** fullvideoname = getFullVideoName();
	
	max_line_len = 1024;
	cout<<"start to generate fisher vector"<<endl;

	char *feat_file_path  = new char[100];

	char *feat_mbh_fv_file = new char[100];

	char *descriptorFilePath = new char[100];
	char *descriptorFileName = new char[100];


	char *gmmMeansMbhFileName = new char[100];
	strcpy(gmmMeansMbhFileName, gmmMeansMbh);
	char *gmmCovariancesMbhFileName = new char[100];
	strcpy(gmmCovariancesMbhFileName, gmmCovariancesMbh);
	char *gmmPriorsMbhFileName = new char[100];
	strcpy(gmmPriorsMbhFileName, gmmPriorsMbh);
	float * Mbhmeans = new float[gmmSize*MBH_DI];
	float * Mbhcovariances = new float[gmmSize*MBH_DI];
	float * Mbhpriors = new float[gmmSize];
	readGmmFromFile(Mbhmeans,gmmMeansMbhFileName);
	readGmmFromFile(Mbhcovariances,gmmCovariancesMbhFileName);
	readGmmFromFile(Mbhpriors,gmmPriorsMbhFileName);

	for (size_t i = 0; i < num_videos; i++) {

		strcpy(feat_file_path,featDir);
		strcpy(feat_mbh_fv_file,strcat(strcat(strcat(feat_file_path,"mbh/"),basename(fullvideoname[i])),"-fv"));
	
		strcpy(descriptorFilePath,descriptor_path);
		strcpy(descriptorFileName,strcat(descriptorFilePath,basename(fullvideoname[i])));

		cout<<descriptorFileName<<endl;
		ifstream files(feat_mbh_fv_file);
		if(!files)
			getAndSaveFV(descriptorFileName, gmmSize,Mbhmeans, Mbhcovariances, Mbhpriors, feat_mbh_fv_file);
		else
		{
			files.close();
			cout<<"file exist!"<<endl;
			continue;
		}
		
		
	}

	releaseFullVideoName(fullvideoname);

	delete [] feat_file_path;
	
	delete [] feat_mbh_fv_file;
	delete [] descriptorFilePath;
	delete [] descriptorFileName;



	delete [] gmmMeansMbhFileName;
	delete [] gmmCovariancesMbhFileName;
	delete [] gmmPriorsMbhFileName;
	delete [] Mbhmeans;
	delete [] Mbhcovariances;
	delete [] Mbhpriors;




	return 0;
}

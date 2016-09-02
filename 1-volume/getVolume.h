#ifndef _GETVOLUME_H
#define _GETVOLUME_H
#include "generial.h"
#include <map>
#include <utility>
using namespace cv;
using namespace std;
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
void  getArea(float coordinatex,float coordinatey,int frameNum,int width, int height,int startFrame1,int endFrame1,int startFrame2,int endFrame2,int startFrame3,int endFrame3, int *i, int *j,int *k,int *q)
{
	if(coordinatex < float(width/2))
	{
		if(coordinatey < float(height/2))
		{
			*i = 1;
		}
		else
			*i = 3;
	}
	else
	{
		if(coordinatey < float(height/2))
		{
			*i = 2;
		}
		else
			*i = 4;
	}
	if(startFrame1 <= frameNum && endFrame1 >= frameNum)
	{
		*j = 1;
	}
	if(startFrame2 <= frameNum && endFrame2 >= frameNum)
	{
		*k = 1;
	}
	if(startFrame3 <= frameNum && endFrame3 >= frameNum)
	{
		*q = 1;
	}
	return ;
}
void getSubVolume(char *descriptorFileName,map<int, vector<int> > & subvolume,int total_frame_num ,int width, int height,
	int startFrame1,int endFrame1,int startFrame2,int endFrame2,int startFrame3,int endFrame3)
{
	printf("%s,%s\n", descriptorFileName,"select subvolume works");
	int nLines = 0;
	max_line_len = 1024;
	iiline = Malloc(char, max_line_len);
	gzFile file = gzopen(descriptorFileName,"r");
	float value;
	vector<float> temp;
	char* feature;
	while (ReadLine(file) != NULL)
	{
		int iToken,j;
		feature = strtok(iiline," \t");
		int frameNum = atoi(feature);
		feature = strtok(NULL," \t"); // get rid of obj and trj
		float coordinatex = atof(feature);
		feature = strtok(NULL," \t"); // get rid of obj and trj
		float coordinatey = atof(feature);
		// if(nLines < 10){
			//cout<<coordinatex<<" , "<<coordinatey<<endl;
			int x = 0,y = 0,z = 0,w = 0 ;
			getArea(coordinatex,coordinatey,frameNum,width,height,
				startFrame1, endFrame1, startFrame2, endFrame2, startFrame3, endFrame3,&x,&y,&z,&w);

			x *= 10;
			if( y == 1)
				subvolume[x+1].push_back(nLines);
			if( z == 1)
				subvolume[x+2].push_back(nLines);
			if( w == 1)
				subvolume[x+3].push_back(nLines);
		// }
		nLines++;
	}

	//printf("%s, %d and dimension is ,%d \n","the end of features",(*mbh).size(),(*mbh)[0].size() );
	free(iiline);
	gzclose(file);
}

void SaveSubVolume(char * subvolumeFilePath,map<int, vector<int> > & subvolume, char * video){

	map<int, vector<int>  >::iterator beg; 
	int sum = 0;
	for (beg = subvolume.begin();beg != subvolume.end();++beg)
	{
		sum += beg->second.size();
	}
	

	float arverageTrj = float(sum / 12); // 12 subvolume.

	cout<<sum<<endl;
	cout<<arverageTrj<<endl;
	int nLines = 0;
	max_line_len = 1024;
	iiline = Malloc(char, max_line_len);
	gzFile file = gzopen(video,"r");
	ofstream subvolumeFile(subvolumeFilePath);
	int lineNum = 0;
	for (beg = subvolume.begin();beg != subvolume.end();++beg)
	{
		if(beg->second.size() > arverageTrj){
			gzrewind(file);
			for(vector<int>::iterator begvec = beg->second.begin();begvec != beg->second.end();++begvec)  
			{
				lineNum = *begvec;
				while (ReadLine(file) != NULL && nLines < lineNum )
				{
						nLines++;
				}
				for (int s = 0; s < max_line_len; ++s)
				{
					if(iiline[s] != '\n')
						subvolumeFile<<iiline[s];
					else
					{
						subvolumeFile<<endl;
						break;
					}
				}
			}
		}
	}
	subvolumeFile.close();
	free(iiline);
	gzclose(file);

}
void getVolume(char **fullvideoname, char * volume_descriptor_path)
{
	for (int w = 0; w < datasetSize; ++w)
	{
		VideoCapture capture;
		char* video = new char[100];
		strcpy(video,fullvideoname[w]);
		//strcat(video,".avi");
		cout<<fullvideoname[w]<<endl;

		char* subvolumeFilePath = new char[100];
		strcpy(subvolumeFilePath,volume_descriptor_path);
		strcat(subvolumeFilePath,basename(fullvideoname[w]));
		cout<<subvolumeFilePath<<endl;
		capture.open(video);

		if(!capture.isOpened()) {
			fprintf(stderr, "Could not initialize capturing..\n");
			return ;
		}
		// get the number of frames in the video
		int frame_num = 0;
		int width = 0, height = 0;
		while(true) {
			Mat frame;
			capture >> frame;

			if(frame.empty())
				break;

			if(frame_num == 0) {
				width = frame.cols;
				height = frame.rows;
			}
			frame_num++;
	    }
		cout<<frame_num+1<<" frame_num is "<<endl;
		cout<< width<<" is width"<<endl;
		cout<< height<<" is height"<<endl;

		//design the subvolume of dense trajectory.
		int timeStep = (frame_num+1)/3;
		
		int startFrame1 = 0;
		int endFrame1 = timeStep;
		int startFrame2 = timeStep - 15;
		int endFrame2 = startFrame2 + timeStep;
		int startFrame3 = endFrame2 - 15;
		int endFrame3 = frame_num;

		map<int, vector<int> > subvolume;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int sum = (i+1)*10+(j+1);
				vector<int> trajectorySequence;
				subvolume.insert(pair<int, vector<int> > (sum, trajectorySequence));
			}
		}
		// subvolume[11].push_back(12);
		// subvolume[11].push_back(12);
		// subvolume[11].push_back(12);
		// subvolume[11].push_back(12);
		// subvolume[11].push_back(12);
		
		//read the descriptor file to get the discriminative subvolume.
		strcpy(video,descriptor_path);
		strcat(video,basename(fullvideoname[w]));
		cout<<video<<endl;
		getSubVolume(video,subvolume,frame_num ,width, height,startFrame1,endFrame1,startFrame2,endFrame2,startFrame3,endFrame3);
		cout<<subvolume.size()<<" is the size "<<endl;
		SaveSubVolume(subvolumeFilePath,subvolume,video);
		delete video;
		delete subvolumeFilePath;

	}
}
#endif

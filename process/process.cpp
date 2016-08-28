#include "generial.h"
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

void saveTrainAndTestFile(char * fileName, float**resultTrainData, float** resultTestData,int trainSize, int testSize)
{
	cout<<fileName<<endl;
	cout<<trainSize<<" is trainSize "<< testSize<<" is test size"<<endl;
	ifstream exis(fileName);//创建目标文件
	if(exis)
	{
		cout<<"file exist!"<<fileName<<endl;
		exis.close();
		return ;
	}
	ofstream out(fileName);//创建目标文件
	for (int i = 0; i < trainSize; ++i)
	{
		out<<i+1<<" ";
		for (int j = 0; j < trainSize; ++j)
		{
			out<< resultTrainData[i][j]<<" ";
		}
		out<<endl;
	}
	for (int i = 0; i < testSize; ++i)
	{
		out<<i+1<<" ";
		for (int j = 0; j < trainSize; ++j)
		{
		    out<<resultTestData[i][j]<<" ";
		}
		out<<endl;
	}
	out.close();
}
void readWFromFile(char * trainVideoName,float * trainW,int index)
{
	ifstream in(trainVideoName);
	for (int i = 0; i < darwinDimension; ++i)
	{
		in>>trainW[index*darwinDimension + i];
	}
	in.close();
	return ;
}
void darWinNormalizedL2(float ** Data, int frames, int di)
{
    float sum = 0;
    for (int i = 0; i < frames; ++i)
    {
        sum = 0;
        for (int j = 0; j < di; ++j)
        {
            sum += fabs(Data[i][j]) *fabs(Data[i][j]);
        }
        sum = sqrt(sum);
        for (int j = 0; j < di; ++j)
        {
            Data[i][j] /= sum;
        }
    }
    return;
}

int main(int argc, char const *argv[])
{

	char ** trainAndTestVideoName = getFullVideoName();
	char **trainVideoName = new char*[trainNum];
	char **testVideoName = new char*[testNum];
	int trainSize = trainNum;
	int testSize = testNum;
	for (int i = 0; i < trainNum; ++i)
	{
		trainVideoName[i] = new char[filePathSize];
		strcpy(trainVideoName[i],darwin_feature);
		strcat(trainVideoName[i], basename(trainAndTestVideoName[i]));
		strcat(trainVideoName[i],"-w");
	}
	//cout<<trainVideoName[0]<<endl;
	for (int i = 0; i < testNum; ++i)
	{
		testVideoName[i] = new char[filePathSize];
		strcpy(testVideoName[i],darwin_feature);
		strcat(testVideoName[i], basename(trainAndTestVideoName[i+trainNum]));
		strcat(testVideoName[i],"-w");
	}
	
	float * trainW = new float[trainSize*darwinDimension];
	
	
	for (int s = 0; s < trainSize; ++s)
	{
		//cout<<s<<" read trainfile"<<endl;
		cout<<s<<" read trainfile"<<endl;
		readWFromFile(trainVideoName[s],trainW,s);	
		
	}
	
	float * testW = new float[testSize*darwinDimension];
	for (int s = 0; s < testSize; ++s)
	{
		//cout<<s<<" read testFile"<<endl;
		readWFromFile(testVideoName[s],testW,s);	
		cout<<s<<" read testFile"<<endl;
		readWFromFile(testVideoName[s],testW,s);	
	}
	
	CvMat *trainData,*trainDataRev,*resultTrainData;
	trainData = cvCreateMat( trainSize, darwinDimension, CV_32FC1);
	trainDataRev = cvCreateMat( darwinDimension, trainSize, CV_32FC1);
	resultTrainData = cvCreateMat( trainSize, trainSize, CV_32FC1);
	cout<<"before init ======================="<<endl;
	cvInitMatHeader( trainData, trainSize, darwinDimension, CV_32FC1, trainW);
	cout<<"before transpose ======================="<<endl;
	cvTranspose(trainData,trainDataRev);
	cout<<"after transpose ======================="<<endl;
	cvMatMulAdd( trainData, trainDataRev, 0, resultTrainData);
	cout<<"after mul ======================="<<endl;
	

	CvMat *testData,*resultTestData;
	testData = cvCreateMat( testSize, darwinDimension, CV_32FC1);
	resultTestData = cvCreateMat( testSize, trainSize, CV_32FC1);
	cvInitMatHeader( testData, testSize, darwinDimension, CV_32FC1, testW);
	cvMatMulAdd( testData, trainDataRev, 0, resultTestData);


	float ** floatTrainData = new float*[trainSize];
	for (int i = 0; i < trainSize; ++i)
	{
		floatTrainData[i] = new float[trainSize];
		for (int j = 0; j < trainSize ; ++j)
		{
			floatTrainData[i][j] = CV_MAT_ELEM(* resultTrainData,float,i,j);
		}
	}
	float ** floatTestData = new float*[testSize];
	for (int i = 0; i < testSize; ++i)
	{
		floatTestData[i] = new float[trainSize];
		for (int j = 0; j < trainSize ; ++j)
		{
			floatTestData[i][j] = CV_MAT_ELEM(* resultTestData,float,i,j);
		}
	}
	
	darWinNormalizedL2(floatTrainData, trainSize, trainSize);
	darWinNormalizedL2(floatTestData, testSize, trainSize);

	saveTrainAndTestFile(trainAndTestFilePath,floatTrainData,floatTestData,trainSize, testSize);
	delete []trainW;

	delete []testW;
	for (int i = 0; i < trainNum; ++i)
	{
		delete trainVideoName[i];
		delete floatTrainData[i];
	}
	delete []floatTrainData;
	delete []trainVideoName;
	
	for (int i = 0; i < testNum; ++i)
	{
		delete testVideoName[i];
		delete floatTestData[i];
	}
	delete []floatTestData;
	delete []testVideoName;
	cvReleaseMat(&trainData);
	cvReleaseMat(&trainDataRev);
	cvReleaseMat(&testData);
	cvReleaseMat(&resultTrainData);
	cvReleaseMat(&resultTestData);
	return 0;
}

#include <iostream>
#include <memory.h>
#include <string.h>
using namespace std;
extern "C"
{
	#include <vl/generic.h>
	#include <vl/vlad.h>
	#include <vl/kmeans.h>
	#include <vl/kdtree.h>
}
int main(int argc, char const *argv[])
{
	/*The VLAD encoding of a set of features is obtained by using the function vl_vlad_encode. 
	The function can be applied to both float or double data types.
	vl_vlad_encode requires a visual dictionary, for example obtained by using K-means clustering. 
	Furthermore, the assignments of features to dictionary elements must be pre-computed, for example by using KD-trees and forests.
	In the following example code, the vocabulary is first created using the KMeans clustering, 
	then the points, that are to be encoded are assigned to its corresponding nearest vocabulary words, 
	after that the original vlad encoding routine without any normalization option takes place. 
	At the end of the process the encoding is stored in the enc variable.*/
	vl_uint32 * indexes;
	float * assignments;
	void * enc;
	
	int numData = 5;
	int dimension = 4;
	int numCenters = 3;
	float **data = new float*[numData];
	for (int i = 0; i < numData; ++i)
	{
		data[i] = new float[dimension];
		for (int j = 0; j < dimension; ++j)
		{
			data[i][j] = (i+j)*10;
			cout<<data[i][j]<<" ";
		}
		cout<<endl;
	}
	int numDataToEncode = 3;
	float **dataToEncode = new float*[numDataToEncode];
	for (int i = 0; i < numDataToEncode; ++i)
	{
		dataToEncode[i] = new float[dimension];
		for (int j = 0; j < dimension; ++j)
		{
			dataToEncode[i][j] = (i+j)*9;
			cout<<dataToEncode[i][j]<<" ";
		}
		cout<<endl;
	}
	// create a KMeans object and run clustering to get vocabulary words (centers)
	VlKMeans * kmeans = vl_kmeans_new (VL_TYPE_FLOAT, VlDistanceL2) ;
	// Use Lloyd algorithm
	vl_kmeans_set_algorithm (kmeans, VlKMeansElkan) ;
	
	vl_kmeans_set_max_num_iterations (kmeans, 100) ;

	vl_kmeans_cluster (kmeans,
	                   data,
	                   dimension,
	                   numData,
	                   numCenters) ;
	// find nearest cliuster centers for the data that should be encoded
	indexes = (vl_uint32 *)vl_malloc(sizeof(vl_uint32) * numDataToEncode);
	float * distances = (float *)vl_malloc(sizeof(float *) * numData) ;
	vl_kmeans_quantize(kmeans,indexes,distances,dataToEncode,numDataToEncode);
	for (int i = 0; i < numDataToEncode; ++i)
	{
		cout<<indexes[i]<<" ";
	}
	cout<<" is the indexes" <<endl;
	
	for (int i = 0; i < numData; ++i)
	{
		cout<<distances[i]<<" ";
	}
	cout<<" is the distances" <<endl;
	// convert indexes array to assignments array,
	// which can be processed by vl_vlad_encode
	assignments = (float *)vl_malloc(sizeof(float) * numDataToEncode * numCenters);
	memset(assignments, 0, sizeof(float) * numDataToEncode * numCenters);
	for(int i = 0; i < numDataToEncode; i++) {
	  assignments[i * numCenters + indexes[i]] = 1.;
	}

	// allocate space for vlad encoding
	enc = vl_malloc(sizeof(VL_TYPE_FLOAT) * dimension * numCenters);
	// do the encoding job
	vl_vlad_encode (enc, VL_TYPE_FLOAT,
	                vl_kmeans_get_centers(kmeans), dimension, numCenters,
	                dataToEncode, numDataToEncode,
	                (void *)assignments,
	                0) ;
	for (int i = 0; i < dimension*numCenters; ++i)
	{
		cout<< ((float * )enc)[i]<<" ";
	}
	cout<<endl;
	return 0;
}
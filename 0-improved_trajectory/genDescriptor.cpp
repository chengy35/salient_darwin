#include "genDescriptors.h"
int main(int argc, char const *argv[])
{
	char **fullvideoname = getFullVideoName();
	genDescriptors(0,num_videos,fullvideoname,descriptor_path);
	releaseFullVideoName(fullvideoname);
	return 0;
}
#include "getVolume.h"
int main(int argc, char const *argv[])
{
	char **fullvideoname = getFullVideoName();
	getVolume(fullvideoname,volume_descriptor_path);
	releaseFullVideoName(fullvideoname);
	return 0;
}
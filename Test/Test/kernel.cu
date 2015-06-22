#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <stdio.h>
float initRI[4][4][4] = { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
{ 1.05, 1.05, 1.1, 1.1, 1.05, 1.05, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 },
{ 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2 },
{ 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.21, 1.22, 1.3, 1.3, 1.19, 1.17 }
};
int mapSeq(void* in, void **out, int size, int length);
int demapSeq(void** in);
int marchPhoton(char* ocTree_d, float* nTree_d, float* direction, char* radiance, float* position, int exp, int num, float scale);
int constructOctree(float *ri, int exp, char **out,float delta);
int gaussianHandle(float* b, int exp);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
__global__ void minmaxKernel(float *o, float *b, int len, int l2, int l1)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= len) return;
	int x, y, z;
	int ox, oy, oz;
	float min, max;
	x = idx / l2;
	y = (idx % l2) / l1;
	z = (idx % l2) % l1;
	ox = (x)* 2;
	oy = (y)* 2;
	oz = (z)* 2;
	//000
	min = o[(ox*l2 * 4 + oy*l1 * 2 + oz) * 2];
	max = o[(ox*l2 * 4 + oy*l1 * 2 + oz) * 2 + 1];
	//100
	if (o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz) * 2] < min) min = o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz) * 2];
	if (o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz) * 2 + 1] > max) max = o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz) * 2 + 1];
	//010
	if (o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2] < min) min = o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2];
	if (o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2 + 1] > max) max = o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2 + 1];
	//001
	if (o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1) * 2] < min) min = o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1) * 2];
	if (o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1) * 2 + 1] > max) max = o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1) * 2 + 1];
	//110
	if (o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2] < min) min = o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2];
	if (o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2 + 1] > max) max = o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz) * 2 + 1];
	//101
	if (o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1) * 2] < min) min = o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1) * 2];
	if (o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1) * 2 + 1] > max) max = o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1) * 2 + 1];
	//011
	if (o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2] < min) min = o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2];
	if (o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2 + 1] > max) max = o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2 + 1];
	//111
	if (o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2] < min) min = o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2];
	if (o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2 + 1] > max) max = o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1) * 2 + 1];
	b[(x*l2 + y*l1 + z) * 2] = min;
	b[(x*l2 + y*l1 + z) * 2 + 1] = max;
}
__global__ void ocTreeKernel(float *m, char *u, char* d, int len, int l2, int l1, int exp, float delta)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= len) return;
	int x, y, z;
	int nx, ny, nz;
	x = idx / l2;
	y = (idx % l2) / l1;
	z = (idx % l2) % l1;
	nx = (x)* 2;
	ny = (y)* 2;
	nz = (z)* 2;
	if (u[x*l2 + y*l1 + z] != 0)
	{
		d[nx*l2 * 4 + ny*l1 * 2 + nz] = u[x*l2 + y*l1 + z]; //000
		d[(nx + 1)*l2 * 4 + ny*l1 * 2 + nz] = u[x*l2 + y*l1 + z]; //100
		d[(nx + 1)*l2 * 4 + (ny + 1)*l1 * 2 + nz] = u[x*l2 + y*l1 + z];//110
		d[(nx + 1)*l2 * 4 + (ny + 1)*l1 * 2 + nz + 1] = u[x*l2 + y*l1 + z];//111
		d[nx*l2 * 4 + (ny + 1)*l1 * 2 + nz] = u[x*l2 + y*l1 + z];//010
		d[nx*l2 * 4 + (ny + 1)*l1 * 2 + nz + 1] = u[x*l2 + y*l1 + z]; //011
		d[nx*l2 * 4 + ny*l1 * 2 + nz + 1] = u[x*l2 + y*l1 + z];//001
		d[(nx + 1)*l2 * 4 + ny*l1 * 2 + nz + 1] = u[x*l2 + y*l1 + z]; //101
		return;
	}
	if (m[(x*l2 + y*l1 + z) * 2 + 1] - m[(x*l2 + y*l1 + z) * 2] < delta)
	{
		d[nx*l2 * 4 + ny*l1 * 2 + nz] = exp; //000
		d[(nx + 1)*l2 * 4 + ny*l1 * 2 + nz] = exp; //100
		d[(nx + 1)*l2 * 4 + (ny + 1)*l1 * 2 + nz] = exp;//110
		d[(nx + 1)*l2 * 4 + (ny + 1)*l1 * 2 + nz + 1] = exp;//111
		d[nx*l2 * 4 + (ny + 1)*l1 * 2 + nz] = exp;//010
		d[nx*l2 * 4 + (ny + 1)*l1 * 2 + nz + 1] = exp; //011
		d[nx*l2 * 4 + ny*l1 * 2 + nz + 1] = exp;//001
		d[(nx + 1)*l2 * 4 + ny*l1 * 2 + nz + 1] = exp; //101
		return;
	}
	if (u[x*l2 + y*l1 + z] == 0)
	{
		d[nx*l2 * 4 + ny*l1 * 2 + nz] = 0; //000
		d[(nx + 1)*l2 * 4 + ny*l1 * 2 + nz] = 0; //100
		d[(nx + 1)*l2 * 4 + (ny + 1)*l1 * 2 + nz] = 0;//110
		d[(nx + 1)*l2 * 4 + (ny + 1)*l1 * 2 + nz + 1] = 0;//111
		d[nx*l2 * 4 + (ny + 1)*l1 * 2 + nz] = 0;//010
		d[nx*l2 * 4 + (ny + 1)*l1 * 2 + nz + 1] = 0; //011
		d[nx*l2 * 4 + ny*l1 * 2 + nz + 1] = 0;//001
		d[(nx + 1)*l2 * 4 + ny*l1 * 2 + nz + 1] = 0; //101
		return;
	}
}
__global__ void marchKernel(char* ocTree, float* nTree, float* direction, float* position, char* radiance, int num, int l2, int l1, float scale)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx > num)	return;
	float3 dir,pos,max,tn;
	uchar3 rad;
	int3 opos;
	int seq,level,len;
	float ratio = 0.0,n=1.0,length;
	rad.x = (unsigned char)radiance[idx * 3];
	rad.y = (unsigned char)radiance[idx * 3 + 1];
	rad.z = (unsigned char)radiance[idx * 3 + 2];
	//dir.x = 1.0*((unsigned char)direction[idx * 3]-127) / 255.0;
	//dir.y = 1.0*((unsigned char)direction[idx * 3 + 1]-127) / 255.0;
	//dir.z = 1.0*((unsigned char)direction[idx * 3 + 2]-127) / 255.0;
	dir.x = direction[idx * 3];
	dir.y = direction[idx * 3 + 1];
	dir.z = direction[idx * 3 + 2];
	pos.x = position[idx * 3] + (l1 / 2)*scale;
	pos.y = position[idx * 3 + 1] + (l1 / 2)*scale;
	pos.z = position[idx * 3 + 2] + (l1 / 2)*scale;
	opos.x = (int)(pos.x/scale);
	opos.y = (int)(pos.y/scale);
	opos.z = (int)(pos.z/scale);
	seq = opos.z*l2 + opos.y*l1 + opos.x;//Need to reconfigure
	level = (unsigned char)ocTree[seq];
	n = nTree[seq];
	tn.x = opos.x > 0 ? nTree[seq] - nTree[seq - 1] : nTree[seq+1] - nTree[seq];
	tn.y = opos.y > 0 ? nTree[seq] - nTree[seq - l1] : nTree[seq + l1] - nTree[seq];
	tn.z = opos.z > 0 ? nTree[seq] - nTree[seq - l2] : nTree[seq + l2] - nTree[seq];
	len = 1 << level;
	max.x = dir.x > 0.0 ? (len*scale - (pos.x - (opos.x - opos.x%len)*scale)) : -(pos.x - (opos.x - opos.x%len)*scale);
	max.y = dir.y > 0.0 ? (len*scale - (pos.y - (opos.y - opos.y%len)*scale)) : -(pos.y - (opos.y - opos.y%len)*scale);
	max.z = dir.z > 0.0 ? (len*scale - (pos.z - (opos.z - opos.z%len)*scale)) : -(pos.z - (opos.z - opos.z%len)*scale);
	ratio = max.x / dir.x;
	if (max.y / dir.y > ratio) ratio = max.y / dir.y;
	if (max.z / dir.z > ratio) ratio = max.z / dir.z;
	length = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z)*ratio;
	position[idx * 3] = position[idx * 3] + dir.x*length / n;
	position[idx * 3 + 1] = position[idx * 3 + 1] + dir.y*length / n;
	position[idx * 3 + 2] = position[idx * 3 + 2] + dir.z*length / n;
	direction[idx * 3] = direction[idx * 3] + length*tn.x;
	direction[idx * 3 + 1] = direction[idx * 3 + 1] + length*tn.y;
	direction[idx * 3 + 2] = direction[idx * 3 + 2] + length*tn.z;
	radiance[idx * 3] = radiance[idx * 3] * exp(-0.0000000095781*length);
	radiance[idx * 3 + 1] = radiance[idx * 3 + 1] + length*tn.y;
	radiance[idx * 3 + 2] = radiance[idx * 3 + 2] + length*tn.z;
	return;
}
__global__ void gaussianKernel(float *dev_b, float *dev_r, int size, int l2, int l1)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= size) return;
	int x, y, z;
	x = idx / l2;
	y = (idx % l2) / l1;
	z = (idx % l2) % l1;
}

int main()
{
	char *ocTree;
	int a = constructOctree((float *)initRI, 2,&ocTree,0.090);
	for (int i = 0; i < 64; i++) printf("%u ", ocTree[i]); printf("\n\n");
	float *in;
	FILE *stream;
	if ((stream = fopen("pos.data", "rb")) == NULL) /* open file TEST.$$$ */
	{
		fprintf(stderr, "Cannot open output file.\n");
		return;
	}
	in = (float *)malloc(125000 * 3 * sizeof(float));
	fread(in, 125000 * 3 * sizeof(float), 1, stream); /* 写的struct文件*/
	fclose(stream); /*关闭文件*/

// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	system("pause");
	free(ocTree);
	return 0;
}

int gaussianHandle(float* b, int exp)
{
	int length = 1 << exp;
	int size = length*length*length;
	float *dev_b, *dev_r;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_b, size*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_r, size*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, size* sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	gaussianKernel <<<(size / 16) + 1, 16 >>>(dev_b, dev_r, size, length*length, length);
	cudaStatus = cudaMemcpy(b, dev_b, size* sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}
	return 0;
Error:
	cudaFree(dev_b);
	cudaFree(dev_r);
	return 1;
}

cudaError_t onePass(float * dev_a, float *b, float** dev_b, int exp){
	int nlength;
	int nsize3, nsize2;
	nlength = 1 << (exp - 1);
	nsize3 = nlength*nlength*nlength;
	nsize2 = nlength*nlength;
	float *dev_o = dev_a;
	float *dev_n = 0;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_n, nsize3 * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	minmaxKernel <<<(nsize3 / 16) + 1, 16 >>>(dev_o, dev_n, nsize3, nsize2, nlength);
	*dev_b = dev_n;
	cudaStatus = cudaMemcpy(b, dev_n, nsize3 * 2 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_n);
	return cudaStatus;
}

int constructOctree(float *ri, int exp,char **out,float delta)
{
	int olength, osize2, osize3;
	float *o_dev, *n_dev, *p_dev;
	char *u_dev = 0, *d_dev;
	float *oRI;
	float **mmPyramid;
	char **ocPyramid;
	olength = 1 << exp;
	osize2 = olength*olength;
	osize3 = olength*olength*olength;
	mmPyramid = (float **)malloc((exp + 1)*sizeof(float *));
	ocPyramid = (char **)malloc((exp + 1)*sizeof(char *));
	int tsize = osize3 * 2;
	for (int i = 0; i <= exp; i++)
	{
		mmPyramid[i] = (float *)malloc(tsize * sizeof(float));
		ocPyramid[i] = (char *)malloc(tsize / 2 * sizeof(char));
		tsize /= 8;
	}
	*out = (char *)malloc(osize3 * sizeof(char));
	oRI = mmPyramid[0];
	for (int i = 0; i < olength; i++)
		for (int j = 0; j < olength; j++)
			for (int k = 0; k < olength; k++)
			{
				*(oRI + (i*osize2 + j*olength + k) * 2) = *(ri + i*osize2 + j*olength + k);
				*(oRI + (i*osize2 + j*olength + k) * 2 + 1) = *(ri + i*osize2 + j*olength + k);
			}

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&o_dev, osize3 * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(o_dev, oRI, osize3 * 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	for (int i = 0; i < exp; i++)
	{
		cudaStatus = onePass(o_dev, mmPyramid[i + 1], &n_dev, exp - i);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "onepass failed!");
			goto Error;
		}
		cudaThreadSynchronize();
		cudaFree(o_dev);
		o_dev = n_dev;
	}
	cudaFree(o_dev);
	ocPyramid[exp][0] = 0;
	cudaStatus = cudaMalloc((void**)&d_dev, 1 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_dev, ocPyramid[exp], 1 * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	for (int i = 0; i <exp; i++){
		olength = 1 << i;
		osize3 = olength*olength*olength;
		cudaStatus = cudaMalloc((void**)&p_dev, osize3 * 2 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(p_dev, mmPyramid[exp - i], osize3 * 2 * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (u_dev)  cudaFree(u_dev);
		u_dev = d_dev;
		cudaStatus = cudaMalloc((void**)&d_dev, osize3 * 8 * sizeof(char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		ocTreeKernel << <(osize3 / 16) + 1, 16 >> >(p_dev, u_dev, d_dev, osize3, olength*olength, olength, exp - i, delta);
		cudaThreadSynchronize();
		cudaStatus = cudaMemcpy(ocPyramid[exp - i - 1], d_dev, osize3 * 8 * sizeof(char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}
/*	for (int i = 0; i < 1; i++) printf("%d ", ocPyramid[2][i]); printf("\n\n");
	for (int i = 0; i < 8; i++) printf("%d ", ocPyramid[1][i]); printf("\n\n");
	for (int i = 0; i < 64; i++) printf("%d ", ocPyramid[0][i]); printf("\n\n");*/
	olength = 1 << exp;
	osize3 = olength*olength*olength;
	memcpy(*out, ocPyramid[0], osize3 * sizeof(char));
	for (int i = 0; i <= exp; i++)
	{
		free(mmPyramid[i]);
		free(ocPyramid[i]);				
	}
Error:
	cudaFree(o_dev);
	cudaFree(n_dev);
	cudaFree(p_dev);
	cudaFree(u_dev);
	cudaFree(d_dev);
	return 0;
}
int mapSeq(void* in, void **out, int size, int length)
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	void *tmp;
	cudaStatus = cudaMalloc((void**)&tmp, length * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(tmp, in, length * size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	*out = tmp;
	return 0;
Error:
	cudaFree(tmp);
	return 1;
}
int demapSeq(void** in)
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaFree(*in);
	return 0;
Error:
	return 1;
}
int marchPhoton(char* ocTree_d,float* nTree_d, float* direction, char* radiance, float* position, int exp, int num,float scale){
	char  *radiance_d;
	float *direction_d, *position_d;
	int len, osize3;
	len = 1 << exp;
	osize3 = len*len*len;

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&direction_d, num *3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(direction_d, direction, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&radiance_d, num * 3 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(radiance_d, radiance, num * 3 * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&position_d, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(position_d, position, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//minmaxKernel << <(nsize3 / 16) + 1, 16 >> >(dev_o, dev_n, nsize3, nsize2, nlength);
	marchKernel <<< (num / 16) + 1, 16 >>> (ocTree_d, nTree_d, direction_d, position_d, radiance_d, num, len*len, len, scale);
	cudaStatus = cudaMemcpy(position, position_d, num * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(direction, direction_d, num * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(radiance, radiance_d, num * 3 * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(radiance_d);
	cudaFree(direction_d);
	cudaFree(position_d);
	return 0;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#define M_PI 3.14159265358979323846
#define M_E 2.718281828459
int pixelCount = 640 * 480;
float eyex = 0.0f;
float eyey = 0.0f;
float eyez = 0.0f;
float *seePos = NULL;
float *seeDir = NULL;
float TableZ = -200.0;
int TableR =160;
float initRI[4][4][4] = { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
{ 1.05, 1.05, 1.1, 1.1, 1.05, 1.05, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 },
{ 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2 },
{ 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.21, 1.22, 1.3, 1.3, 1.19, 1.17 }
};
unsigned char test[4][4][4][2];
int mapSeq(float* in, float **out, int size, int length);
int demapSeq(float** in);
int marchPhoton(unsigned char* ocTree_d, float* nTree_d, float* direction, unsigned char* radiance, float* position, int exp, int num, float scale);
int constructOctree(unsigned char *ri, int exp, unsigned char **out, float delta);
int gaussianHandle(float* b, int exp);
int marchPhoton(unsigned char* ocTree, unsigned char* nTree_c, float* direction, float* radiance,
	float* photondir, float* photonrad, float* photonpos, int exp, int num, float scale,
	float** Table, int tableSize, float tableScale,float tmpn);
int collectPhoton(unsigned char* ocTree, unsigned char* nTree_c, float* direction, float* radiance,
	float* photondir, float* photonrad, float* photonpos, int exp, int num, float scale,
	int **o_offset, int  **o_tableOffset, int **o_flag,
	float p1x, float p1y, float p1z, float p2x, float p2y, float p2z,
	float p3x, float p3y, float p3z, float p4x, float p4y, float p4z,
	float tmpn);


__global__ void minmaxKernel(unsigned int *o, unsigned int *b, int len, int l2, int l1)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= len) return;
	int x, y, z;
	int ox, oy, oz;
	unsigned int min = 0, max = 0;
	x = idx / l2;
	y = (idx % l2) / l1;
	z = (idx % l2) % l1;
	ox = (x)* 2;
	oy = (y)* 2;
	oz = (z)* 2;
	//000
	//min = o[len];
	min = o[(ox*l2 * 4 + oy*l1 * 2 + oz)] & 0xffff;
	max = (o[(ox*l2 * 4 + oy*l1 * 2 + oz)] >> 16) & 0xffff;
	//100
	if ((o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz)] & 0xffff) < min) min = o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz)] & 0xffff;
	if (((o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz)] >> 16) & 0xffff)> max) max = (o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz)] >> 16) & 0xffff;
	//010
	if ((o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz)] & 0xffff) < min) min = o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz)] & 0xffff;
	if (((o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz)] >> 16) & 0xffff) > max) max = (o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz)] >> 16) & 0xffff;
	//001
	if ((o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1)] & 0xffff) < min) min = o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1)] & 0xffff;
	if (((o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1)] >> 16) & 0xffff)> max) max = (o[(ox*l2 * 4 + oy*l1 * 2 + oz + 1)] >> 16) & 0xffff;
	//110
	if ((o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz)] & 0xffff)< min) min = o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz)] & 0xffff;
	if (((o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz)] >> 16) & 0xffff) > max) max = (o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz)] >> 16) & 0xffff;
	//101
	if ((o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1)] & 0xffff) < min) min = o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1)] & 0xffff;
	if (((o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1)] >> 16) & 0xffff) > max) max = (o[((ox + 1)*l2 * 4 + oy*l1 * 2 + oz + 1)] >> 16) & 0xffff;
	//011
	if ((o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] & 0xffff) < min) min = o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] & 0xffff;
	if (((o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] >> 16) & 0xffff)> max) max = (o[(ox*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] >> 16) & 0xffff;
	//111
	if ((o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] & 0xffff) < min) min = o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] & 0xffff;
	if (((o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] >> 16) & 0xffff)> max) max = (o[((ox + 1)*l2 * 4 + (oy + 1)*l1 * 2 + oz + 1)] >> 16) & 0xffff;
	b[idx] = (min & 0xffff) + ((max << 16) & 0xffff0000);
}
__global__ void ocTreeKernel(unsigned int *m, unsigned char *u, unsigned char* d, int len, int l2, int l1, int exp, float delta)
{
	int id_x = blockIdx.x*blockDim.x + threadIdx.x;
	int id_y = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = id_x + id_y*gridDim.x*blockDim.x;
	if (idx >= len) return;
	int x, y, z;
	int nx, ny, nz;
	unsigned int min, max;
	x = idx / l2;
	y = (idx % l2) / l1;
	z = (idx % l2) % l1;
	nx = (x)* 2;
	ny = (y)* 2;
	nz = (z)* 2;
	min = m[(x*l2 + y*l1 + z)] & 0xffff;
	max = (m[(x*l2 + y*l1 + z)] >> 16) & 0xffff;
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
	if ((max - min)*0.0003 < delta)
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
__global__ void marchKernel(unsigned char* ocTree, float* nTree, float* radiance, float*direction, float* d1, float* d2,
	float* r1, float* r2, float* p1, float *p2, int num, int l2, int l1, float scale, float step, int* leftPhoto,
	float* table, int tableSize, float tableScale, float tableZ,float tmpn)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= num)	return;
	float3 dir, pos, max, tn, rad, np, nd, nr, pstep, rstep, dstep;
	int3 opos;
	int seq, level, len, testid = -1;
	float ratio = 0.0, n = 1.0, length, half = (l1 / 2)*scale;
	rad.x = r1[idx * 3];
	rad.y = r1[idx * 3 + 1];
	rad.z = r1[idx * 3 + 2];
	pos.x = p1[idx * 3] + half;
	pos.y = p1[idx * 3 + 1] + half;
	pos.z = p1[idx * 3 + 2] + half;
	opos.x = (int)(pos.x / scale);
	opos.y = (int)(pos.y / scale);
	opos.z = (int)(pos.z / scale);
	if (idx == testid)
		printf("pos %lf %lf %lf\n", pos.x - half, pos.y - half, pos.z - half);
	if ((rad.x + rad.y + rad.z < 0.000001)
		|| (opos.x < 0 || opos.x >= l1 || opos.y < 0 || opos.y >= l1 || opos.z < 0 || opos.z >= l1))
	{
		p2[idx * 3] = pos.x - half;
		p2[idx * 3 + 1] = pos.y - half;
		p2[idx * 3 + 2] = pos.z - half;
		d2[idx * 3] = d1[idx * 3];
		d2[idx * 3 + 1] = d1[idx * 3 + 1];
		d2[idx * 3 + 2] = d1[idx * 3 + 2];
		r2[idx * 3] = rad.x;
		r2[idx * 3 + 1] = rad.y;
		r2[idx * 3 + 2] = rad.z;
		return;
	}
	atomicAdd(leftPhoto, 1);
	dir.x = d1[idx * 3];
	dir.y = d1[idx * 3 + 1];
	dir.z = d1[idx * 3 + 2];
	if (idx == testid)
		printf("dir %lf %lf %lf\n", dir.x, dir.y, dir.z);
	seq = (l1 - opos.z - 1)*l2 + opos.y*l1 + opos.x;//Need to reconfigur
	level = (unsigned char)ocTree[seq];
	n = nTree[seq];
	if (n < 0.01) n = 1.0;
	if (idx == testid)
		printf("n %lf\n", n);
	if (idx == testid)
		printf("level %d\n", level);
	float value = -1, tlength;
	int tseq, l3 = l1*l1*l1;
	
	if (idx == testid)
		printf("value %lf tlength %lf tn %lf %lf %lf\n", value, tlength, tn.x, tn.y, tn.z);
	
	len = 1 << level;
	max.x = dir.x > 0.0 ? (len*scale - (pos.x - (opos.x - opos.x%len)*scale)) : -(pos.x - (opos.x - opos.x%len)*scale);
	max.y = dir.y > 0.0 ? (len*scale - (pos.y - (opos.y - opos.y%len)*scale)) : -(pos.y - (opos.y - opos.y%len)*scale);
	max.z = dir.z > 0.0 ? (len*scale - (pos.z - (opos.z - opos.z%len)*scale)) : -(pos.z - (opos.z - opos.z%len)*scale);
	ratio = max.x / dir.x;
	if (max.y / dir.y < ratio) ratio = max.y / dir.y;
	if (max.z / dir.z < ratio) ratio = max.z / dir.z;
	length = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z)*ratio*1.010;
	if (length < scale) length = scale;
	if (idx == testid)
		printf("limit %lf %lf %lf\n", (opos.x - opos.x%len)*scale - half, (opos.y - opos.y%len)*scale - half, (opos.z - opos.z%len)*scale - half);
	if (idx == testid)
		printf("max.x.y.z %lf %lf %lf\n", max.x, max.y, max.z);
	if (idx == testid)
		printf("ratio %lf\n", ratio);
	if (idx == testid)
		printf("length %lf\n", length);
	np.x = pos.x - half + dir.x*length / n;
	np.y = pos.y - half + dir.y*length / n;
	np.z = pos.z - half + dir.z*length / n;
	if (tlength > 120.0 )
		n = 1.0;
	else
		n = 1.2;
	tlength = sqrt((pos.x - half)*(pos.x - half) + (pos.y - half)*(pos.y - half) + (pos.z - half)*(pos.z - half));
	float ntlength = sqrt(np.x*np.x + np.y*np.y + np.z*np.z);
	if (ntlength <= 120.0&&tlength >= 120.0 || ntlength >= 120.0&&tlength <= 120.0)
	{
		tn.x = -((pos.x - half) / tlength)*tmpn / length;
		tn.y = -((pos.y - half) / tlength)*tmpn / length;
		tn.z = -((pos.z - half) / tlength)*tmpn / length;
	}
	else
	{
		tn.x = tn.y = tn.z = 0;
	}
	nd.x = dir.x + length*tn.x;
	nd.y = dir.y + length*tn.y;
	nd.z = dir.z + length*tn.z;
	ratio = exp(-0.00000095781*length);
	nr.x = rad.x*ratio;
	nr.y = rad.y*ratio;
	nr.z = rad.z*ratio;
	p2[idx * 3] = np.x;
	p2[idx * 3 + 1] = np.y;
	p2[idx * 3 + 2] = np.z;
	d2[idx * 3] = nd.x;
	d2[idx * 3 + 1] = nd.y;
	d2[idx * 3 + 2] = nd.z;
	r2[idx * 3] = rad.x*ratio;
	r2[idx * 3 + 1] = rad.y*ratio;
	r2[idx * 3 + 2] = rad.z*ratio;
	float Max, length_1, length_2;
	Max = dir.x>0 ? dir.x : -dir.x;
	Max = dir.y > 0 ? (dir.y > Max ? dir.y : Max) : (-dir.y > Max ? -dir.y : Max);
	Max = dir.z > 0 ? (dir.z > Max ? dir.z : Max) : (-dir.z > Max ? -dir.z : Max);
	pstep.x = dir.x / Max*scale;
	pstep.y = dir.y / Max*scale;
	pstep.z = dir.z / Max*scale;
	opos.x = (int)(pos.x / scale);
	opos.y = (int)(pos.y / scale);
	opos.z = (int)(pos.z / scale);
	length_1 = sqrt(pstep.x*pstep.x + pstep.y*pstep.y + pstep.z*pstep.z);
	int size = int(length / length_1);
	rstep.x = (nr.x - rad.x) / size;
	rstep.y = (nr.y - rad.y) / size;
	rstep.z = (nr.z - rad.z) / size;
	length_1 = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
	length_2 = sqrt(nd.x*nd.x + nd.y*nd.y + nd.z*nd.z);
	dir.x = dir.x / length_1;
	dir.y = dir.y / length_1;
	dir.z = dir.z / length_1;
	dstep.x = (nd.x / length_2 - dir.x / length_1) / size;
	dstep.y = (nd.y / length_2 - dir.y / length_1) / size;
	dstep.z = (nd.z / length_2 - dir.z / length_1) / size;
	float3 tr, td;
	float ar_1, ar_2;
	for (int i = 0; i < size; i++)
	{
		opos.x = (int)((pos.x) / scale);
		opos.y = (int)((pos.y) / scale);
		opos.z = (int)((pos.z) / scale);
		seq = (l1 - 1 - opos.z)*l2 + opos.y*l1 + opos.x;
		tr.x = radiance[seq * 3];
		tr.y = radiance[seq * 3 + 1];
		tr.z = radiance[seq * 3 + 2];
		td.x = direction[seq * 3];
		td.y = direction[seq * 3 + 1];
		td.z = direction[seq * 3 + 2];
		ar_1 = tr.x + tr.y + tr.z;
		ar_2 = rad.x + rad.y + rad.z;
		nd.x = td.x*ar_1 + dir.x*ar_2;
		nd.y = td.y*ar_1 + dir.y*ar_2;
		nd.z = td.z*ar_1 + dir.z*ar_2;
		length_1 = sqrt(nd.x*nd.x + nd.y*nd.y + nd.z*nd.z);
		nd.x = nd.x / length_1;
		nd.y = nd.y / length_1;
		nd.z = nd.z / length_1;
		atomicAdd(&radiance[seq * 3], rad.x);
		atomicAdd(&radiance[seq * 3 + 1], rad.y);
		atomicAdd(&radiance[seq * 3 + 2], rad.z);
		atomicAdd(&direction[seq * 3], nd.x - td.x);
		atomicAdd(&direction[seq * 3 + 1], nd.y - td.y);
		atomicAdd(&direction[seq * 3 + 2], nd.z - td.z);
		pos.x = pos.x + pstep.x;
		pos.y = pos.y + pstep.y;
		pos.z = pos.z + pstep.z;
		rad.x = rad.x + rstep.x;
		rad.y = rad.y + rstep.y;
		rad.z = rad.z + rstep.z;
		dir.x = dir.x + dstep.x;
		dir.y = dir.y + dstep.y;
		dir.z = dir.z + dstep.z;
	}
	int x, y;
	double halft = 1.0*(tableSize / 2)*tableScale;
	if ((pos.z - half) <= tableZ )
	{
		if (dir.z <= -0.0001)
		{
			ratio = (tableZ - (pos.z - half)) / dir.z;
			pos.x = pos.x + dir.x*ratio;
			pos.y = pos.y + dir.y*ratio;
			x = int((pos.x - half + halft) / tableScale);
			y = int((pos.y - half + halft) / tableScale);
			if (x >= 0 && x < tableSize&&y >= 0 && y <= tableSize)
			{
				atomicAdd(&table[(y*tableSize + x) * 3], rad.x);
				atomicAdd(&table[(y*tableSize + x) * 3 + 1], rad.y);
				atomicAdd(&table[(y*tableSize + x) * 3 + 2], rad.z);
				r2[idx * 3] = 0.0f;
				r2[idx * 3 + 1] = 0.0f;
				r2[idx * 3 + 2] = 0.0f;
			}

		}
	}
	return;
}
__global__ void gaussianKernel(float *dev_o, float *dev_n, float *dev_r, int size, int l2, int l1, int ksize)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= size) return;
	int x, y, z, hsize = ksize / 2, tseq;
	float tmp = 0, origin = dev_o[idx], tsum = 0;
	x = idx / l2;
	y = (idx % l2) / l1;
	z = (idx % l2) % l1;
	for (int i = -hsize; i <= hsize; i++)
		for (int j = -hsize; j <= hsize; j++)
			for (int k = -hsize; k <= hsize; k++)
			{
		tseq = idx + i*l2 + j*l1 + k;
		if (tseq < 0 || tseq >= size)
			tmp = origin;
		else
			tmp = dev_o[tseq];
		tsum += tmp*dev_r[(i + hsize)*l2 + (j + hsize)*l1 + k];
			}
	dev_n[idx] = tsum;
	return;
}
__global__ void gaussian2DKernel(float *dev_o, float *dev_n, float *dev_r, int size, int l1, int ksize, float scale)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= size) return;
	int x, y, z, hsize = ksize / 2, tseq;
	float3 tmp = make_float3(0.0f, 0.0f, 0.0f);
	float3 tsum = make_float3(0.0f, 0.0f, 0.0f);
	float3  origin = make_float3(dev_o[idx * 3], dev_o[idx * 3 + 1], dev_o[idx * 3 + 2]);
	float cal;
	x = idx % l1;
	y = idx/ l1;
	for (int i = -hsize; i <= hsize; i++)
		for (int j = -hsize; j <= hsize; j++)
		{
			tseq = idx + i*l1 + j;
			if (tseq < 0 || tseq >= size)
				tmp = origin;
			else
			{
				tmp.x = dev_o[tseq * 3];
				tmp.y = dev_o[tseq * 3+1];
				tmp.z = dev_o[tseq * 3+2];
			}				
			cal = dev_r[(i + hsize)*ksize + j + hsize];
			tsum.x = tsum.x + cal*tmp.x;
			tsum.y = tsum.y + cal*tmp.y;
			tsum.z = tsum.z + cal*tmp.z;
		}
	dev_n[idx * 3] = tsum.x*scale;
	dev_n[idx * 3 + 1] = tsum.y*scale;
	dev_n[idx * 3 + 2] = tsum.z*scale;
	return;
}
void gaussian2D(float* texture_d, float** textureout,int len,int kernelsize,float sigma,float scale){
	float *cal;
	float *texture_nd,*cal_d;
	int r = (kernelsize / 2);
	cal = (float*)malloc(kernelsize*kernelsize*sizeof(float));
	float sum = 0;
	for (int i = -r; i <= r; i++){
		for (int j = -r; j <=r; j++){
			cal[(i + r)*kernelsize + (j + r)] = (1.0f / (pow(2 * M_PI*sigma*sigma, 1)))*pow(M_E, -(i*i + j*j) / (2 * pow(sigma, 2.0)));
			sum += cal[(i + r)*kernelsize + (j + r)];
		}
	}
	for (int i = -r; i <=r; i++){
		for (int j = -r; j <=r; j++){
			cal[(i + r)*kernelsize + (j + r)] /= (float)(sum);
		}
	}
	cudaError_t cudaStatus = cudaMalloc((void**)&cal_d, kernelsize*kernelsize* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(cal_d, cal, kernelsize*kernelsize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&texture_nd, len*len*3*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	int size = len*len;
	gaussian2DKernel <<<(size / 256) + 1, 256 >>>(texture_d, texture_nd, cal_d, size, len, kernelsize,scale);
	
	*textureout = (float*)malloc(len*len*3*sizeof(float));
	cudaStatus = cudaMemcpy(*textureout, texture_nd, len*len * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	if (texture_d) cudaFree(texture_d);
	if (cal_d) cudaFree(cal_d);
	if (texture_nd) cudaFree(texture_nd);
}
cudaError_t onePass(unsigned int * dev_a, unsigned int *b, unsigned int** dev_b, int exp){
	int nlength;
	int nsize3, nsize2;
	nlength = 1 << (exp - 1);
	nsize3 = nlength*nlength*nlength;
	nsize2 = nlength*nlength;
	unsigned int *dev_o = dev_a;
	unsigned int *dev_n = 0;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_n, nsize3* sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	minmaxKernel << <(nsize3 / 512) + 1, 512 >> >(dev_o, dev_n, nsize3, nsize2, nlength);
	cudaThreadSynchronize();
	*dev_b = dev_n;
	cudaStatus = cudaMemcpy(b, dev_n, nsize3 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	return cudaStatus;
Error:
	cudaFree(dev_n);
	return cudaStatus;
}

int constructOctree(unsigned char *ri, int exp, unsigned char **out, float delta)
{
	int olength, osize2, osize3;
	unsigned int *o_dev = 0, *n_dev = 0, *p_dev = 0;
	unsigned char *u_dev = 0, *d_dev = 0;
	unsigned int *oRI;
	unsigned int **mmPyramid;
	unsigned char **ocPyramid;
	olength = 1 << exp;
	osize2 = olength*olength;
	osize3 = olength*olength*olength;
	mmPyramid = (unsigned int **)malloc((exp + 1)*sizeof(unsigned int *));
	ocPyramid = (unsigned char **)malloc((exp + 1)*sizeof(unsigned char *));
	int tsize = osize3;
	for (int i = 0; i <= exp; i++)
	{
		mmPyramid[i] = (unsigned int *)malloc(tsize * sizeof(unsigned int));
		ocPyramid[i] = (unsigned char *)malloc(tsize * sizeof(unsigned char));
		if (!mmPyramid[i] || !ocPyramid[i]){
			fprintf(stderr, "CPU memory Malloc failed!");
			goto Error;
		}
		tsize /= 8;
	}
	*out = (unsigned char *)malloc(osize3 * sizeof(unsigned char));
	if (!out){
		fprintf(stderr, "CPU memory Malloc failed!");
		goto Error;
	}
	oRI = mmPyramid[0];
	unsigned int c1, c2, tmp;
	for (int i = 0; i < olength; i++)
		for (int j = 0; j < olength; j++)
			for (int k = 0; k < olength; k++)
			{
		c1 = (ri[(i*osize2 + j*olength + k) * 2]) & 0xff;
		c2 = (ri[(i*osize2 + j*olength + k) * 2 + 1]) & 0xff;
		tmp = ((c1 << 8) & 0xff00) + (c2 & 0xff);
		oRI[(i*osize2 + j*olength + k)] = (tmp << 16) + tmp;
			}

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&o_dev, osize3* sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(o_dev, oRI, osize3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
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
	cudaStatus = cudaMalloc((void**)&d_dev, 1 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_dev, ocPyramid[exp], 1 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	for (int i = 0; i <exp; i++){
		olength = 1 << i;
		osize3 = olength*olength*olength;
		cudaStatus = cudaMalloc((void**)&p_dev, osize3 * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(p_dev, mmPyramid[exp - i], osize3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (u_dev)  cudaFree(u_dev);
		u_dev = d_dev;
		d_dev = 0;
		cudaStatus = cudaMalloc((void**)&d_dev, osize3 * 8 * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		dim3 gridD;
		gridD.x = ((osize3 / 256) + 1) >= 65536 ? 65535 : ((osize3 / 256) + 1);
		gridD.y = ((((osize3 / 256) + 1) / 65536) + 1);
		ocTreeKernel << <gridD, dim3(16, 16) >> >(p_dev, u_dev, d_dev, osize3, olength*olength, olength, exp - i, delta);
		cudaThreadSynchronize();
		cudaStatus = cudaMemcpy(ocPyramid[exp - i - 1], d_dev, osize3 * 8 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaFree(p_dev);
	}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}
	olength = 1 << exp;
	osize3 = olength*olength*olength;
	memcpy(*out, ocPyramid[0], osize3 * sizeof(unsigned char));
	for (int i = 0; i <= exp; i++)
	{
		free(mmPyramid[i]);
		free(ocPyramid[i]);
	}
Error:
	if (o_dev) cudaFree(o_dev);
	if (n_dev) cudaFree(n_dev);
	if (p_dev) cudaFree(p_dev);
	if (u_dev) cudaFree(u_dev);
	if (d_dev) cudaFree(d_dev);
	return 0;
}
int mapSeq(float* in, float **out, int size, int length)
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	float *tmp = 0;
	cudaStatus = cudaMalloc((float**)&tmp, length * size);
	if (!tmp)
	{
		fprintf(stderr, "CPU memory Malloc failed!");
		goto Error;
	}
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
	if (tmp) cudaFree(tmp);
	return 1;
}
int demapSeq(float** in)
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
int marchPhoton(unsigned char* ocTree, unsigned char* nTree_c, float* direction, float* radiance,
	float* photondir, float* photonrad, float* photonpos, int exp, int num, float scale,
	float** Table, int tableSize, float tableScale,float tmpn){
	float *photondir_do = 0, *photonrad_do = 0, *photonpos_do = 0;
	float *photondir_dn = 0, *photonrad_dn = 0, *photonpos_dn = 0;
	unsigned char * ocTree_d;
	float *table_d, *table;
	float *nTree_d, *direction_d, *radiance_d;
	float *nTree;
	int len, *leftPhoton_d, osize3, len2;
	len = 1 << exp;
	len2 = len*len;
	osize3 = len*len*len;
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	nTree = (float *)malloc(osize3*sizeof(float));
	if (!nTree) goto Error;
	for (int i = 0; i < osize3; i++)
	{
		nTree[i] = (((nTree_c[i * 2] << 8) & 0xff00) + (nTree_c[i * 2 + 1] & 0xff))*0.0003;
	}
	cudaStatus = cudaMalloc((void**)&nTree_d, osize3* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(nTree_d, nTree, osize3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&ocTree_d, osize3*sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(ocTree_d, ocTree, osize3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	memset(direction, 0, osize3 * 3 * sizeof(float));
	cudaStatus = cudaMalloc((void**)&direction_d, osize3 * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(direction_d, direction, osize3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	memset(radiance, 0, osize3 * 3 * sizeof(float));
	cudaStatus = cudaMalloc((void**)&radiance_d, osize3 * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(radiance_d, radiance, osize3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&leftPhoton_d, 1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&photondir_dn, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(photondir_dn, photondir, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&photonrad_dn, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(photonrad_dn, photonrad, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&photonpos_dn, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(photonpos_dn, photonpos, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//----------------------------------------------------------------
	table = (float *)malloc(tableSize * tableSize * 3 * sizeof(float));
	cudaStatus = cudaMalloc((void**)&table_d, tableSize * tableSize * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(table_d, 0, tableSize * tableSize * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	//---------------------------------------------------------
	int *leftPhoton;
	leftPhoton = (int *)malloc(sizeof(int));
	*leftPhoton = num;
	while (*leftPhoton > num*0.001)
	{
		*leftPhoton = 0;
		cudaStatus = cudaMemcpy(leftPhoton_d, leftPhoton, 1 * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		if (photondir_do)cudaFree(photondir_do);
		if (photonrad_do)cudaFree(photonrad_do);
		if (photonpos_do)cudaFree(photonpos_do);
		photondir_do = photondir_dn;
		photonrad_do = photonrad_dn;
		photonpos_do = photonpos_dn;
		cudaStatus = cudaMalloc((void**)&photonpos_dn, num * 3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&photonrad_dn, num * 3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&photonpos_dn, num * 3 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		marchKernel << <(num / 256) + 1, 256 >> >(ocTree_d, nTree_d, radiance_d, direction_d, photondir_do, photondir_dn,
			photonrad_do, photonrad_dn, photonpos_do, photonpos_dn, num, len2, len, scale, -1, leftPhoton_d,
			table_d, tableSize, tableScale, TableZ,tmpn);
		cudaThreadSynchronize();
		cudaStatus = cudaMemcpy(leftPhoton, leftPhoton_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		printf("leftphoto %d\n\n", *leftPhoton);
	}
	cudaStatus = cudaMemcpy(radiance, radiance_d, osize3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(direction, direction_d, osize3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	*Table = table_d;
Error:
	if (radiance_d) cudaFree(radiance_d);
	if (direction_d) cudaFree(direction_d);
	if (photondir_do) cudaFree(photondir_do);
	if (photondir_dn) cudaFree(photondir_dn);
	if (photonrad_do) cudaFree(photonrad_do);
	if (photonrad_dn) cudaFree(photonrad_dn);
	if (photonpos_do) cudaFree(photonpos_do);
	if (photonpos_dn) cudaFree(photonpos_dn);
	if (ocTree_d) cudaFree(ocTree_d);
	if (nTree_d) cudaFree(nTree_d);
	if (nTree) free(nTree);
	return 0;
}
//

__global__ void collectKernel(unsigned char* ocTree, float* nTree, float* radiance, float*direction, float* d1, float* d2,
	float* r1, float* r2, float* p1, float *p2, int num, int l2, int l1, float scale, float step, int* leftPhoto,float tmpn,int* flag)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= num)	return;
	float xscale = 2.0;
	float3 dir, pos, max, tn, rad, np, nd, nr, pstep, rstep, dstep;
	int3 opos;
	int seq, level, len;
	float ratio = 0.0, n = 1.0, length, half = (l1 / 2)*scale;
	rad.x = r1[idx * 3];
	rad.y = r1[idx * 3 + 1];
	rad.z = r1[idx * 3 + 2];
	pos.x = p1[idx * 3] + half;
	pos.y = p1[idx * 3 + 1] + half;
	pos.z = p1[idx * 3 + 2] + half;
	opos.x = (int)(pos.x / scale);
	opos.y = (int)(pos.y / scale);
	opos.z = (int)(pos.z / scale);
	if ((opos.x < 0 || opos.x >= l1 || opos.y < 0 || opos.y >= l1 || opos.z < 0 || opos.z >= l1))
	{
		p2[idx * 3] = pos.x - half;
		p2[idx * 3 + 1] = pos.y - half;
		p2[idx * 3 + 2] = pos.z - half;
		d2[idx * 3] = d1[idx * 3];
		d2[idx * 3 + 1] = d1[idx * 3 + 1];
		d2[idx * 3 + 2] = d1[idx * 3 + 2];
		r2[idx * 3] = rad.x;
		r2[idx * 3 + 1] = rad.y;
		r2[idx * 3 + 2] = rad.z;
		return;
	}
	atomicAdd(leftPhoto, 1);
	dir.x = d1[idx * 3];
	dir.y = d1[idx * 3 + 1];
	dir.z = d1[idx * 3 + 2];
	seq = (l1 - opos.z - 1)*l2 + opos.y*l1 + opos.x;//Need to reconfigur
	n = nTree[seq];
	if (n < 0.01) n = 1.0;
	float tlength,ntlength;
	length = scale / xscale;
	tlength = sqrt((pos.x - half)*(pos.x - half) + (pos.y - half)*(pos.y - half) + (pos.z - half)*(pos.z - half));	
	np.x = pos.x - half + dir.x*length / n;
	np.y = pos.y - half + dir.y*length / n;
	np.z = pos.z - half + dir.z*length / n;
	ntlength = sqrt(np.x*np.x + np.y*np.y + np.z*np.z);
	if (ntlength <= 120.0&&tlength >= 120.0 || ntlength >=120.0&&tlength <= 120.0)
	{
		tn.x = -((pos.x - half) / tlength)*tmpn / length;
		tn.y = -((pos.y - half) / tlength)*tmpn / length;
		tn.z = -((pos.z - half) / tlength)*tmpn / length;
		flag[idx] = 3;
	}
	else
	{
		tn.x = tn.y = tn.z = 0;
	}
	nd.x = dir.x + length*tn.x;
	nd.y = dir.y + length*tn.y;
	nd.z = dir.z + length*tn.z;

	float ar_1, ar_2;
	float cosr = 0.0f;
	float dr = 0.0f;
	float dg = 0.0f;
	float db = 0.0f;

	float3 tr, td;
	float xscale3 = xscale*xscale*xscale;

	for (int i = 0; i < 1; i++)
	{
		opos.x = (int)((pos.x) / scale);
		opos.y = (int)((pos.y) / scale);
		opos.z = (int)((pos.z) / scale);
		seq = (l1 - 1 - opos.z)*l2 + opos.y*l1 + opos.x;

		tr.x = radiance[seq * 3];
		tr.y = radiance[seq * 3 + 1];
		tr.z = radiance[seq * 3 + 2];

		td.x = direction[seq * 3];
		td.y = direction[seq * 3 + 1];
		td.z = direction[seq * 3 + 2];

		cosr = dir.x * td.x + dir.y * td.y + dir.z * td.z;
		cosr /= sqrt(td.x * td.x + td.y * td.y + td.z * td.z);
		cosr /= sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

		if (cosr < 0){		
			dr = -cosr * radiance[seq * 3] * (1 / (xscale3));
			dg = -cosr * radiance[seq * 3 + 1] * (1 / (xscale3));
			db = -cosr * radiance[seq * 3 + 2] * (1 / (xscale3));
		}
		else{
			dr = 0.0;
			dg = 0.0;
			db = 0.0;
		}

	}

	p2[idx * 3] = np.x;
	p2[idx * 3 + 1] = np.y;
	p2[idx * 3 + 2] = np.z;

	d2[idx * 3] = nd.x;
	d2[idx * 3 + 1] = nd.y;
	d2[idx * 3 + 2] = nd.z;

	r2[idx * 3] = rad.x + dr;
	r2[idx * 3 + 1] = rad.y + dg;
	r2[idx * 3 + 2] = rad.z + db;

	return;
}

inline __global__ void RayTriangleIntersection(float * pos, float * dir, int num, float* t,
	int *flag, const float3 v0, const float3 edge1, const float3 edge2, int nflag)
{

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= num)	return;

	float3 rayPos;
	rayPos.x = pos[idx * 3];
	rayPos.y = pos[idx * 3 + 1];
	rayPos.z = pos[idx * 3 + 2];

	float3 rayDir;
	rayDir.x = dir[idx * 3];
	rayDir.y = dir[idx * 3 + 1];
	rayDir.z = dir[idx * 3 + 2];

	float3 tvec = rayPos - v0;
	float3 pvec = cross(rayDir, edge2);
	float  det = dot(edge1, pvec);
	det = __fdividef(1.0f, det);

	float u = dot(tvec, pvec) * det;


	if (u < 0.0f || u > 1.0f){
		return;
	}

	float3 qvec = cross(tvec, edge1);

	float v = dot(rayDir, qvec) * det;


	if (v < 0.0f || (u + v) > 1.0f){
		return;
	}
	float tt = dot(edge2, qvec) * det;
	if (tt <= 0 && nflag == 2 && rayDir.z>0)
		return;
	float3 val =tt* rayDir + rayPos;
	t[idx * 3] = val.x;
	t[idx * 3 + 1] = val.y;
	t[idx * 3 + 2] = val.z;
	flag[idx] = nflag/* + flag[idx]*/;
	return;
}

__global__ void toTableOffset(int num, float* pos, int * offset, float p1x, float p1y, float p1z,
	float p2x, float p2y, float p2z, float p3x, float p3y, float p3z, int x, int y, int *flag, int nflag){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num)	return;	
	if (flag[idx] != nflag) return;
	float3 ori = make_float3(p1x, p1y, p1z);
	float3 xd = make_float3(p2x - p1x, p2y - p1y, p2z - p1z);
	float3 yd = make_float3(p3x - p1x, p3y - p1y, p3z - p1z);
	float3 photo = make_float3(pos[idx * 3], pos[idx * 3 + 1], pos[idx * 3 + 2]);
	float3 photod = make_float3(photo.x - ori.x, photo.y - ori.y, photo.z - ori.z);
	float l1 = sqrt(xd.x*xd.x + xd.y*xd.y + xd.z*xd.z);
	float l2 = sqrt(yd.x*yd.x + yd.y*yd.y + yd.z*yd.z);
	float c1 = dot(photod, xd);
	float c2 = dot(photod, yd);
	c1 /= l1;
	c2 /= l2;
	int nx = int(c1/l1*x);
	int ny = int(c2/l2*y);
	offset[idx] = ny*x + nx;
}
int collectPhoton(unsigned char* ocTree, unsigned char* nTree_c, float* direction, float* radiance,
	float* photondir, float* photonrad, float* photonpos, int exp, int num, float scale,
	int **o_offset,int  **o_tableOffset,int **o_flag,
	float p1x, float p1y, float p1z, float p2x, float p2y, float p2z,
	float p3x, float p3y, float p3z ,float p4x, float p4y, float p4z,
	float tmpn){
	float *photondir_do = 0, *photonrad_do = 0, *photonpos_do = 0;
	float *photondir_dn = 0, *photonrad_dn = 0, *photonpos_dn = 0;

	unsigned char * ocTree_d;
	float *nTree_d, *direction_d, *radiance_d;
	float *nTree;
	int len, *leftPhoton_d, osize3, len2;
	len = 1 << exp;
	len2 = len*len;
	osize3 = len*len*len;
	cudaError_t cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	nTree = (float *)malloc(osize3*sizeof(float));
	if (!nTree) goto Error;
	for (int i = 0; i < osize3; i++)
	{
		nTree[i] = (((nTree_c[i * 2] << 8) & 0xff00) + (nTree_c[i * 2 + 1] & 0xff))*0.0003;
	}

	cudaStatus = cudaMalloc((void**)&nTree_d, osize3* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(nTree_d, nTree, osize3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------

	cudaStatus = cudaMalloc((void**)&direction_d, osize3 * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(direction_d, direction, osize3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------

	cudaStatus = cudaMalloc((void**)&radiance_d, osize3 * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(radiance_d, radiance, osize3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&leftPhoton_d, 1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&photondir_dn, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(photondir_dn, photondir, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&photonrad_dn, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(photonrad_dn, photonrad, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&photonpos_dn, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(photonpos_dn, photonpos, num * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//---------------------------------------------------------
	cudaStatus = cudaMalloc((void**)&photonpos_do, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&photondir_do, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&photonrad_do, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//----------------------------------------------------------------
	int *flag_d;
	
	cudaStatus = cudaMalloc((void**)&flag_d, num * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(flag_d, -1, num  * sizeof(int));
	//--------------------
	int *leftPhoton;
	leftPhoton = (int *)malloc(sizeof(int));
	*leftPhoton = num;

	float * posTmp;
	float * radTmp;
	float * dirTmp;

	while (*leftPhoton > num*0.001)
	{
		*leftPhoton = 0;
		cudaStatus = cudaMemcpy(leftPhoton_d, leftPhoton, 1 * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		posTmp = photonpos_do;
		radTmp = photonrad_do;
		dirTmp = photondir_do;

		photondir_do = photondir_dn;
		photonrad_do = photonrad_dn;
		photonpos_do = photonpos_dn;

		photondir_dn = dirTmp;
		photonrad_dn = radTmp;
		photonpos_dn = posTmp;

		collectKernel << <(num / 256) + 1, 256 >> >(NULL, nTree_d, radiance_d, direction_d, photondir_do, photondir_dn,
			photonrad_do, photonrad_dn, photonpos_do, photonpos_dn, num, len2, len, scale, -1, leftPhoton_d,tmpn,flag_d);
		cudaThreadSynchronize();
		cudaStatus = cudaMemcpy(leftPhoton, leftPhoton_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, cudaGetErrorString(cudaStatus));
			goto Error;
		}
		printf("leftphoto %d\n\n", *leftPhoton);
	}

	float * intersection, *intersectionTable;
	
	
	cudaStatus = cudaMalloc((void**)&intersection, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&intersectionTable, num * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(intersection, 0, num * 3 * sizeof(float));
	cudaStatus = cudaMemset(intersectionTable, 0, num * 3 * sizeof(float));

	RayTriangleIntersection << <(num / 256) + 1, 256 >> >(photonpos_dn, photondir_dn, num, intersection, flag_d,
		make_float3(p1x, p1y, p1z),
		make_float3(p2x-p1x, p2y-p1y, p2z-p1z),
		make_float3(p4x-p1x, p4y-p1y, p4z-p1z), 1);
	cudaThreadSynchronize();
	RayTriangleIntersection << <(num / 256) + 1, 256 >> >(photonpos_dn, photondir_dn, num, intersection, flag_d,
		make_float3(p3x,p3y,p3z),
		make_float3(p4x-p3x, p4y-p3y, p4z-p3z),
		make_float3(p2x-p3x, p2y-p3y, p2z-p3z), 1);
	cudaThreadSynchronize();
	RayTriangleIntersection << <(num / 256) + 1, 256 >> >(photonpos_dn, photondir_dn, num, intersectionTable, flag_d,
		make_float3(TableR, TableR, TableZ),
		make_float3(-2 * TableR, 0.0f, 0.0f),
		make_float3(0.0f, -2 * TableR, 0.0f), 2);
	cudaThreadSynchronize();
	RayTriangleIntersection << <(num / 256) + 1, 256 >> >(photonpos_dn, photondir_dn, num, intersectionTable, flag_d,
		make_float3(-TableR, -TableR, TableZ),
		make_float3(+2 * TableR, 0.0f, 0.0f),
		make_float3(0.0f, 2 * TableR, 0.0f), 2);
	cudaThreadSynchronize();
	int *tex, *tableTex_d;// = (int *)cudaMalloc(num * sizeof(int));

	cudaStatus = cudaMalloc((void**)&tex, num * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&tableTex_d, num * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(tex, 0, num  * sizeof(int));
	cudaStatus = cudaMemset(tableTex_d, 0, num  * sizeof(int));
	toTableOffset << <(num / 256) + 1, 256 >> >(num, intersection, tex,
		p4x, p4y, p4z,
		p3x, p3y, p3z,
		p1x, p1y, p1z,
		512, 512, flag_d, 1);
	cudaThreadSynchronize();
	toTableOffset << <(num / 256) + 1, 256 >> >(num, intersectionTable, tableTex_d,
		-TableR, -TableR, TableZ,
		TableR, -TableR, TableZ,
		-TableR, TableR, TableZ,
		TableR * 2, TableR*2, flag_d, 2);
	cudaThreadSynchronize();
	cudaStatus = cudaMemcpy(photonrad, photonrad_dn, num * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int * memTex = (int *)malloc(num*sizeof(int));
	int * tableTex = (int *)malloc(num*sizeof(int));
	int * flag = (int *)malloc(num*sizeof(int));
	cudaStatus = cudaMemcpy(memTex, tex, num * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(tableTex, tableTex_d, num * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(flag, flag_d, num * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	*o_offset = memTex;
	*o_tableOffset = tableTex;
	*o_flag = flag;
Error:
	if (radiance_d) cudaFree(radiance_d);
	if (direction_d) cudaFree(direction_d);
	if (photondir_do) cudaFree(photondir_do);
	if (photondir_dn) cudaFree(photondir_dn);
	if (photonrad_do) cudaFree(photonrad_do);
	if (photonrad_dn) cudaFree(photonrad_dn);
	if (photonpos_do) cudaFree(photonpos_do);
	if (photonpos_dn) cudaFree(photonpos_dn);
	if (nTree_d) cudaFree(nTree_d);
	if (nTree) free(nTree);
	if (intersection) cudaFree(intersection);
	if (intersectionTable) cudaFree(intersectionTable);
	if (tableTex_d) cudaFree(tableTex_d);
	if (flag_d) cudaFree(flag_d);
	return 0;
}

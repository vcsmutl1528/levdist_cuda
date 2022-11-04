
#define STRLEN 64

#define LEVENSHTEIN_MAX_LENTH 255

#define LEVDROWSIZE (LEVENSHTEIN_MAX_LENTH+1)

struct levdist_in {
	int l1, l2;
	int r;
	long long int ts, te;
	char s1[STRLEN];
	char s2[STRLEN];
};

struct levdist_out {
	int r;
	long long int ts, te;
};

extern "C" __global__ void levdist (levdist_in *lvd_in, levdist_out *lvd_out)
{
	int *p1, *p2, *tmp;
	int i1, i2, c0, c1, c2;
	int l1 = lvd_in->l1, l2 = lvd_in->l2;

	if (l1==0) { lvd_out[threadIdx.x].r = l2; return; }
	if (l2==0) { lvd_out[threadIdx.x].r = l1; return; }

	if (l1>LEVENSHTEIN_MAX_LENTH || l2>LEVENSHTEIN_MAX_LENTH) {
		lvd_out[threadIdx.x].r = -1;
		return;
	}
	
	int levdp1a [LEVDROWSIZE];
	int levdp2a [LEVDROWSIZE];

	p1 = levdp1a;
	p2 = levdp2a;

	for (i2=0; i2<=l2; i2++)
		p1 [i2] = i2;

	for (i1=0; i1<l1; i1++)
	{
		p2 [0] = p1 [0] + 1;
		for (i2=0; i2<l2; i2++)
		{
			c0 = p1 [i2] + (lvd_in->s1[i1] == lvd_in->s2[i2] ? 0 : 1);
			c1 = p1 [i2+1] + 1;
			if (c1<c0) c0 = c1;
			c2 = p2 [i2] + 1;
			if (c2<c0) c0 = c2;
			p2 [i2+1] = c0;
		}
		tmp=p1; p1=p2; p2=tmp;
	}

//	c0 = p1 [l2];

	lvd_out [threadIdx.x].r = c0;
}

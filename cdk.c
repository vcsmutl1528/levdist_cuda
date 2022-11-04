
#include <stdio.h>
#include <stdlib.h>

#include <windows.h>

#include <cuda.h>

static const char pszSourceFile[] = __FILE__;
BOOL fFirstMsg = TRUE;

#define CU_assert(val) CU_assert_f (val, __LINE__)
#define CU_warning(val) CU_warning_f (val, __LINE__)

void CU_throw_e (CUresult result, const char *file, unsigned line);
void CU_warning_msg (CUresult result, const char *file, unsigned line);

inline void CU_assert_f (CUresult result, unsigned int line) {
	if (result != CUDA_SUCCESS)
		CU_throw_e (result, pszSourceFile, line);
}

inline void CU_warning_f (CUresult result, unsigned int line) {
	if (result != CUDA_SUCCESS)
		CU_warning_msg (result, pszSourceFile, line);
}

int exception_filter (unsigned int code, struct _EXCEPTION_POINTERS *ep);

char szDeviceName [256];

#define STRLEN 64

#define NUM_THREADS 256

struct levdist_in {
	int l1, l2;
	char s1 [STRLEN];
	char s2 [STRLEN];
} levdist_in;

struct levdist_out {
	int r;
	long long int ts, te;
} levdist_out [NUM_THREADS];

int __cdecl main (int argc, const char *argv[])
{
	int devId, n;
	int a, b, i;
	CUdevice cuDev;
	CUmodule cuModule;
	CUcontext cuCtx;
	CUfunction cuFunc;
	CUdeviceptr cuDevPtrIn, cuDevPtrOut;
	BOOL fCtxCreated = FALSE;
	BOOL fModuleLoaded = FALSE;
	BOOL fMemInAllocd = FALSE;
	BOOL fMemOutAllocd = FALSE;

	__try { __try {
		CU_assert (cuInit (0));
		CU_assert (cuDeviceGetCount (&n));
		if (n == 0) {
			puts ("No CUDA devices found.");
			return EXIT_FAILURE;
		} else
			printf ("%d CUDA capable GPU device(s) detected.\n", n);
		devId = 0;
		CU_assert (cuDeviceGetName (szDeviceName, sizeof (szDeviceName), devId));
		szDeviceName [sizeof (szDeviceName)-1] = '\0';
		printf ("CUDA Device: %s\n", szDeviceName);
		CU_assert (cuDeviceGet (&cuDev, devId));
		CU_assert (cuDeviceGetAttribute (&n, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev));
		printf ("Compute mode = %d\n", n);
		if (n == CU_COMPUTEMODE_PROHIBITED) {
			puts ("Error: The device is currently in prohibited mode.");
			return EXIT_FAILURE;
		}
		CU_assert (cuDeviceGetAttribute (&a, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDev));
		CU_assert (cuDeviceGetAttribute (&b, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDev));
		printf ("Compute capability: %d.%d\n", a, b);

		CU_assert (cuCtxCreate (&cuCtx, 0, cuDev));
		fCtxCreated = TRUE;
		CU_assert (cuCtxSetCurrent (cuCtx));

		CU_assert (cuModuleLoad (&cuModule, "levdist.cubin"));
		fModuleLoaded = TRUE;

		for (i=0; i < 32; i++) {
			levdist_in.s1 [i*2 + 0] = i + 1;
			levdist_in.s1 [i*2 + 1] = i + 1;
			levdist_in.s2 [STRLEN - i*2 - 1] = i + 1;
			levdist_in.s2 [STRLEN - i*2 - 2] = i + 1;
		}
		levdist_in.l1 = levdist_in.l2 = STRLEN;

		CU_assert (cuModuleGetFunction (&cuFunc, cuModule, "levdist"));

		CU_assert (cuMemAlloc (&cuDevPtrIn, sizeof (struct levdist_in)));
		fMemInAllocd = TRUE;
		CU_assert (cuMemAlloc (&cuDevPtrOut, sizeof (struct levdist_out)));
		fMemOutAllocd = TRUE;
		CU_assert (cuMemcpyHtoD (cuDevPtrIn, &levdist_in, sizeof (levdist_in)));

		void *arr[] = { &cuDevPtrIn, &cuDevPtrOut };
		CU_assert (cuLaunchKernel (cuFunc, 1, 1, 1, NUM_THREADS, 1, 1, 0, 0,
			&arr[0], 0));
		CU_assert (cuCtxSynchronize ());
		CU_assert (cuMemcpyDtoH (&levdist_out, cuDevPtrOut, sizeof (levdist_out)));

		CU_warning (cuMemFree (cuDevPtrOut));
		fMemOutAllocd = FALSE;
		CU_warning (cuMemFree (cuDevPtrIn));
		fMemInAllocd = FALSE;
		CU_warning (cuModuleUnload (cuModule));
		fModuleLoaded = FALSE;
		CU_warning (cuCtxDestroy (cuCtx));
		fCtxCreated = FALSE;

	} __except (exception_filter (GetExceptionCode(), GetExceptionInformation())) {
		fprintf (stderr, "Exception 0x%08x. Exiting.\n", GetExceptionCode ());
	} } __finally {
		if (fCtxCreated) CU_warning (cuCtxDestroy (cuCtx));
		if (fModuleLoaded) CU_warning (cuModuleUnload (cuModule));
		puts ("Finally.");
	}
	return EXIT_SUCCESS;
}

int exception_filter(unsigned int code, struct _EXCEPTION_POINTERS *ep)
{
	return code & 0x20000000 ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH;
}

static void CU_result_msg (CUresult result, const char *file, unsigned line, const char *severity);

void CU_throw_e (CUresult result, const char *file, unsigned line)
{
	CU_result_msg (result, file, line, "Error");
	RaiseException (0xE0000000 | result & 0xFFFF, 0, 0, 0);
}

void CU_warning_msg (CUresult result, const char *file, unsigned line)
{
	CU_result_msg (result, file, line, "Warning");
}

void CU_result_msg (CUresult result, const char *file, unsigned line, const char *severity)
{
	const char *pszErrName, *pszErrString;

	cuGetErrorName (result, &pszErrName);
	cuGetErrorString (result, &pszErrString);

	if (fFirstMsg) {
		fprintf (stderr, "%s: Timestamp: %s\n", file, __TIMESTAMP__);
		fFirstMsg = FALSE;
	}

	fprintf (stderr, "%s(%d): %s %s (%d): %s.\n", file, line, severity,
		pszErrName, result, pszErrString);
}

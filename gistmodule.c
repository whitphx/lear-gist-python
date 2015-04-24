#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "lear_gist-1.2/gist.h"

static PyObject* gist_extract(PyObject *self, PyObject *args)
{
	int nblocks=4;
	int n_scale=3;
	int orientations_per_scale[50]={8,8,4};
	PyArrayObject *image, *descriptor;

	if (!PyArg_ParseTuple(args, "O", &image))
	{
		return NULL;
	}

	if (PyArray_TYPE(image) != NPY_UINT8) {
		PyErr_SetString(PyExc_TypeError, "type of image must be uint8");
		return NULL;
	}

	if (PyArray_NDIM(image) != 3) {
		PyErr_SetString(PyExc_TypeError, "dimensions of image must be 3.");
		return NULL;
	}

	npy_intp *dims_image = PyArray_DIMS(image);


	const int w = (int) *(dims_image+1);
	const int h = (int) *(dims_image);

	// Read image to color_image_t structure
	color_image_t *im=color_image_new(w,h);

	for (int y=0, i=0 ; y<h ; ++y) {
		for (int x=0 ; x<w ; ++x, ++i) {
			im->c1[i] = *(unsigned char *)PyArray_GETPTR3(image, y, x, 0);
			im->c2[i] = *(unsigned char *)PyArray_GETPTR3(image, y, x, 1);
			im->c3[i] = *(unsigned char *)PyArray_GETPTR3(image, y, x, 2);
		}
	}

	float *desc=color_gist_scaletab(im,nblocks,n_scale,orientations_per_scale);

	int descsize=0;
	/* compute descriptor size */
	for(int i=0;i<n_scale;i++) 
		descsize+=nblocks*nblocks*orientations_per_scale[i];

	descsize*=3; /* color */


	// Allocate output
	npy_intp dim_desc[1] = {descsize};
	descriptor = (PyArrayObject *) PyArray_SimpleNew(1, dim_desc, NPY_FLOAT);

	// Set val
	for (int i=0 ; i<descsize ; ++i) {
		*(float *)PyArray_GETPTR1(descriptor, i) = desc[i];
	}

	return PyArray_Return(descriptor);
}

static PyMethodDef methods[] = {
	{"extract", gist_extract, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initgist(void)
{
    (void)Py_InitModule("gist", methods);
    import_array();
}
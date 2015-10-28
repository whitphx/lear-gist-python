#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "lear_gist-1.2/gist.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

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

	// Extract descriptor
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

	// Release memory
	color_image_delete(im);
	free(desc);

	return PyArray_Return(descriptor);
}

static PyMethodDef gist_methods[] = {
	{"extract", gist_extract, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int gist_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int gist_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "gist",
        NULL,
        sizeof(struct module_state),
        gist_methods,
        NULL,
        gist_traverse,
        gist_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_gist(void)

#else
#define INITERROR return

PyMODINIT_FUNC
initgist(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("gist", gist_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("gist.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

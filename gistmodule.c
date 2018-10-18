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

static PyObject* gist_extract(PyObject *self, PyObject *args, PyObject *keywds)
{
	int nblocks=4;
	int n_scale_default=3;
	int orientations_per_scale_default[3]={8,8,4};
	int n_scale;
	int *orientations_per_scale = NULL;
	PyArrayObject *image, *descriptor;
	PyObject* pyobj_orientations_per_scale = NULL;

	static char *kwlist[] = {"", "nblocks", "orientations_per_scale", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|iO", kwlist,
										&PyArray_Type, &image, &nblocks, &pyobj_orientations_per_scale))
	{
		return NULL;
	}

	// Check validity of image argument
	if (PyArray_TYPE(image) != NPY_UINT8) {
		PyErr_SetString(PyExc_ValueError, "type of image must be uint8");
		return NULL;
	}

	if (PyArray_NDIM(image) != 3) {
		PyErr_SetString(PyExc_ValueError, "dimensions of image must be 3.");
		return NULL;
	}

	// Parse orientations_per_scale argument
	// Ref: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch16s03.html
	if (pyobj_orientations_per_scale != NULL) {
		pyobj_orientations_per_scale = PySequence_Fast(pyobj_orientations_per_scale, "orientations_per_scale must be iterable");
		if (!pyobj_orientations_per_scale) {
			return NULL;
		}

		n_scale = PySequence_Fast_GET_SIZE(pyobj_orientations_per_scale);
		orientations_per_scale = malloc(n_scale * sizeof(int));
		if (!orientations_per_scale) {
			Py_DECREF(pyobj_orientations_per_scale);
			return PyErr_NoMemory();
		}

		for (int i = 0; i < n_scale; ++i) {
			PyObject *long_item;
			PyObject *item = PySequence_Fast_GET_ITEM(pyobj_orientations_per_scale, i);
			if (!item) {
				Py_DECREF(pyobj_orientations_per_scale);
				free(orientations_per_scale);
				return NULL;
			}
			long_item = PyNumber_Long(item);
			if (!long_item) {
				Py_DECREF(pyobj_orientations_per_scale);
				free(orientations_per_scale);
				PyErr_SetString(PyExc_TypeError, "all items of orientations_per_scale must be int");
				return NULL;
			}
			orientations_per_scale[i] = (int) PyLong_AsLong(long_item);  // XXX: Down cast
			Py_DECREF(long_item);
		}

		Py_DECREF(pyobj_orientations_per_scale);
	} else {
		n_scale = n_scale_default;
		orientations_per_scale = orientations_per_scale_default;
	}

	npy_intp *dims_image = PyArray_DIMS(image);

	const int w = (int) *(dims_image+1);
	const int h = (int) *(dims_image);
	const int channels = (int) *(dims_image+2);

	if (w == 0 || h == 0) {
		PyErr_SetString(PyExc_ValueError, "invalid image size.");
		return NULL;
	}
	if (channels != 3) {
		PyErr_SetString(PyExc_ValueError, "invalid color channels.");
		return NULL;
	}

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
	float *desc=color_gist_scaletab(im, nblocks, n_scale, orientations_per_scale);

	/* compute descriptor size */
	int descsize=0;
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
	{"extract", gist_extract, METH_VARARGS | METH_KEYWORDS, "Extracts Lear's GIST descriptor"},
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

#define BTF_IMPLEMENTATION
#include <Python.h>

#include "btf.hh"

static void FreeBTF_py(PyObject *raw_btf) {
  DestroyBTF((BTF *)PyCapsule_GetPointer(raw_btf, NULL));
}

static PyObject *LoadBTF_py(PyObject *self, PyObject *args) {
  PyObject *raw_btf = Py_None;
  char *filename;

  if (PyArg_ParseTuple(args, "s", &filename)) {
    auto *btf = LoadBTF((char *)filename);
    if (btf) {
      raw_btf = PyCapsule_New(btf, NULL, FreeBTF_py);
      // raw_btf = PyCapsule_New(btf, filename, FreeBTF_py);
    } else {
      PyErr_SetString(PyExc_RuntimeError, "cannot read file");
      return NULL;
    }
  }
  return raw_btf;
}

static PyObject *SniffBTF_py(PyObject *self, PyObject *args) {
  PyObject *raw_btf = Py_None;
  PyObject *img_shape = Py_None;
  PyObject *result = Py_None;
  BTF *btf = nullptr;

  if (PyArg_ParseTuple(args, "O", &raw_btf)) {
    if (PyCapsule_IsValid(raw_btf, NULL)) {
      btf = (BTF *)(PyCapsule_GetPointer(raw_btf, NULL));
    } else {
      PyErr_SetString(PyExc_ValueError, "invalid PyCapsule");
      return NULL;
    }
    if (btf) {
      // TODO: null check
      PyObject *view_vecs = PyList_New(btf->ViewCount);
      PyObject *light_vecs = PyList_New(btf->LightCount);

      for (uint32_t view_idx = 0; view_idx < btf->ViewCount; ++view_idx) {
        auto view = btf->Views[view_idx];
        PyList_SET_ITEM(view_vecs, view_idx,
                        Py_BuildValue("(fff)", view.x, view.y, view.z));
      }

      for (uint32_t light_idx = 0; light_idx < btf->ViewCount; ++light_idx) {
        auto light = btf->Lights[light_idx];
        PyList_SET_ITEM(light_vecs, light_idx,
                        Py_BuildValue("(fff)", light.x, light.y, light.z));
      }

      // TODO: int check
      img_shape =
          Py_BuildValue("(iii)", btf->Width, btf->Height, btf->ChannelCount);

      // TODO: int check
      result = Py_BuildValue("(OOO)", view_vecs, light_vecs, img_shape);
    } else {
      PyErr_SetString(PyExc_ValueError, "invalid pointer");
      return NULL;
    }
  }
  return result;
}

static PyObject *FetchBTF_py(PyObject *self, PyObject *args) {
  PyObject *raw_btf = Py_None;
  PyObject *result = Py_None;
  BTF *btf = nullptr;
  uint32_t light_idx, view_idx;

  if (PyArg_ParseTuple(args, "Oii", &raw_btf, &light_idx, &view_idx)) {
    if (PyCapsule_IsValid(raw_btf, NULL)) {
      btf = (BTF *)(PyCapsule_GetPointer(raw_btf, NULL));
    } else {
      PyErr_SetString(PyExc_ValueError, "invalid PyCapsule");
      return NULL;
    }
    if (btf) {
      // TODO: null check
      PyObject *list = PyList_New(btf->Height*btf->Width*3);
      for (uint32_t btf_y = 0; btf_y < btf->Height; ++btf_y) {
        for (uint32_t btf_x = 0; btf_x < btf->Width; ++btf_x) {
          auto spec = BTFFetchSpectrum(btf, light_idx, view_idx, btf_x, btf_y);
          uint32_t btf_xy = (btf_x + btf_y*btf->Width)*3;
          PyList_SET_ITEM(list, btf_xy  , Py_BuildValue("f", spec.x));
          PyList_SET_ITEM(list, btf_xy+1, Py_BuildValue("f", spec.y));
          PyList_SET_ITEM(list, btf_xy+2, Py_BuildValue("f", spec.z));
        }
      }

      result = Py_BuildValue("O", list);
      Py_DECREF(list);
    } else {
      PyErr_SetString(PyExc_ValueError, "invalid pointer");
      return NULL;
    }
  }
  return result;
}

static PyObject *FetchBTF_pixel_py(PyObject *self, PyObject *args) {
  PyObject *raw_btf = Py_None;
  PyObject *result = Py_None;
  BTF *btf = nullptr;
  uint32_t light_idx, view_idx, x_idx, y_idx;

  if (PyArg_ParseTuple(args, "Oiiii", &raw_btf, &light_idx, &view_idx, &x_idx,
                       &y_idx)) {
    if (PyCapsule_IsValid(raw_btf, NULL)) {
      btf = (BTF *)(PyCapsule_GetPointer(raw_btf, NULL));
    } else {
      PyErr_SetString(PyExc_ValueError, "invalid PyCapsule");
      return NULL;
    }
    if (btf) {
      // TODO: null check
      auto spec = BTFFetchSpectrum(btf, light_idx, view_idx, x_idx, y_idx);

      // TODO: int check
      result = Py_BuildValue("(fff)", spec.x, spec.y, spec.z);
    } else {
      PyErr_SetString(PyExc_ValueError, "invalid pointer");
      return NULL;
    }
  }
  return result;
}

static PyMethodDef ubo2014_module_methods[] = {
    {"LoadBTF", (PyCFunction)LoadBTF_py, METH_VARARGS,
     PyDoc_STR("Args: str\n"
               "Return: PyCapsule")},
    {"SniffBTF", (PyCFunction)SniffBTF_py, METH_VARARGS,
     PyDoc_STR("Args: PyCapsule\n"
               "Return: list[tuple[float, float, float]], list[tuple[float, "
               "float, float]]")},
    {"FetchBTF", (PyCFunction)FetchBTF_py, METH_VARARGS,
     PyDoc_STR("Args: PyCapsule, int, int\n"
               "Return: list[list[tuple[float, float, float]]]")},
    {"FetchBTF_pixel", (PyCFunction)FetchBTF_pixel_py, METH_VARARGS,
     PyDoc_STR("Args: PyCapsule, int, int, int, int\n"
               "Return: tuple[float, float, float]")},

    // Sentinel
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef ubo2014_definition = {
    PyModuleDef_HEAD_INIT,

    // name of module
    "ubo2014 cpp_extension",

    // module documentation, may be NULL
    "A Python wrapper of btf.hh by zeroeffects/btf.",

    // size of per-interpreter state of the module,
    // or -1 if the module keeps state in global variables.
    -1,

    ubo2014_module_methods};

PyMODINIT_FUNC PyInit_ubo2014_cpp(void) {
  Py_Initialize();
  return PyModule_Create(&ubo2014_definition);
}

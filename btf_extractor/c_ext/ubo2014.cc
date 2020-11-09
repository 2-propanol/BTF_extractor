#define BTF_IMPLEMENTATION
#include <Python.h>
#include "btf.hh"

static PyObject *LoadBTF_py(PyObject *self, PyObject *args)
{
    PyObject *result = Py_None;
    char *filename;
    uint32_t light_idx, view_idx;

    if (PyArg_ParseTuple(args, "sii", &filename, &light_idx, &view_idx))
    {
        auto *btf = LoadBTF((char *)filename);
        if (btf)
        {
            // TODO: null check
            PyObject *nested_list = PyList_New(btf->Height);

            for (uint32_t btf_y = 0; btf_y < btf->Height; ++btf_y)
            {
                PyObject *list = PyList_New(btf->Width);
                for (uint32_t btf_x = 0; btf_x < btf->Width; ++btf_x)
                {
                    auto spec = BTFFetchSpectrum(btf, light_idx, view_idx, btf_x, btf_y);
                    PyList_SetItem(list, btf_x, Py_BuildValue("(fff)", spec.x, spec.y, spec.z));
                }
                PyList_SetItem(nested_list, btf_y, list);
            }

            // TODO: int check
            result = Py_BuildValue("O", nested_list);

            DestroyBTF(btf);
        }
    }

    return result;
}

static PyMethodDef ubo2014_module_methods[] = {
    {"LoadBTF", (PyCFunction)LoadBTF_py, METH_VARARGS,
     PyDoc_STR(
         "Args: str, int, int\n"
         "Return: list[list[tuple[float, float, float]")},

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

PyMODINIT_FUNC PyInit_ubo2014_cpp(void)
{
    Py_Initialize();
    return PyModule_Create(&ubo2014_definition);
}

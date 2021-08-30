import os
import pickle
import IPython


def load_var_from_ipython(var_name):
    ipython_var = os.path.join(IPython.paths.locate_profile(), "db", "autorestore", var_name)
    var_value = pickle.load(open(ipython_var, 'rb'))
    return var_value

def list_vars_from_ipython():
    ipython_var = os.path.join(IPython.paths.locate_profile(), "db", "autorestore")
    return list(os.listdir(ipython_var))
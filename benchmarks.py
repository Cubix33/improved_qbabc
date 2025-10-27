def cec_benchmark_function(func_name, dim=30):
    # Import inside function to avoid top-level import issues
    from opfunu.cec_based.cec2014 import F12014  # Import explicitly or dynamically
    
    # Dynamically get the class from cec2014 module
    cec2014_module = __import__('opfunu.cec_based.cec2014', fromlist=[func_name])
    func_class = getattr(cec2014_module, func_name)
    return func_class(ndim=dim)
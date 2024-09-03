library(reticulate)

python_version <- r"(C:\Users\PC0118\AppData\Local\Programs\Python\Python311)"
use_python(python_version, required=TRUE)


source_python("DeviceAnalysis.py")
source_python("DeviceEngine.py")
source_python("DeviceProcSettings.py")
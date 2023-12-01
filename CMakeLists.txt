### setup project ###
cmake_minimum_required(VERSION 3.14)

project(gsf LANGUAGES C Fortran)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter NumPy)

# Find OpenMP
find_package(OpenMP REQUIRED)

# The CMake FindPython module does not give us the F2PY include directory
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Generate the C wrapper for the Fortran code
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_twobodymodule.c"
  COMMAND ${Python_EXECUTABLE} -m "numpy.f2py" -m "_twobody" "${CMAKE_SOURCE_DIR}/gsf/twobody.f95"
)

# Add the library target that builds the C wrapper
add_library(_twobody MODULE
    "${CMAKE_CURRENT_BINARY_DIR}/_twobodymodule.c"
    "${F2PY_INCLUDE_DIR}/fortranobject.c"
    "${CMAKE_SOURCE_DIR}/gsf/twobody.f95")

# Link all required libraries
target_link_libraries(_twobody PUBLIC Python::NumPy OpenMP::OpenMP_C)
target_include_directories(_twobody PUBLIC
                           ${F2PY_INCLUDE_DIR}
                           )

# Avoid the "lib" prefix on Linux
set_target_properties(_twobody PROPERTIES PREFIX "")

# Install the library
install(TARGETS _twobody DESTINATION .)

# Write a feature summary in the terminal
include(FeatureSummary)
feature_summary(WHAT ALL)

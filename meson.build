project(
  'scikit-learn',
  'c', 'cpp', 'cython',
  version: run_command('sklearn/_build_utils/version.py', check: true).stdout().strip(),
  license: 'BSD-3',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c11',
    'cpp_std=c++14',
  ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cython = meson.get_compiler('cython')

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.0')
    error('scikit-learn requires GCC >= 8.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('scikit-learn requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

# Read the environment variable, default to an empty string if not set
custom_cflags = meson.get_external_property('C_COMPILER_FLAGS', '')

# Initialize an empty list for custom flags
custom_cflags_list = []

# Split the custom flags string into a list of individual flags, only if it's non-empty
if custom_cflags != ''
    custom_cflags_list = custom_cflags.split()
endif

default_c_args = [
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation'
]

# Combine the default arguments with the custom flags
all_c_args = default_c_args + custom_cflags_list

message('All c args: ' + all_c_args)

# Pass the combined list to the function
_global_c_args = cc.get_supported_arguments(all_c_args)

add_project_arguments(_global_c_args, language : 'c')

message('Custom compile flags from CUSTOM_CFLAGS: ' + custom_cflags)
message('Final C compiler arguments: ' + ' '.join(_global_c_args))

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for scikit-learn). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

tempita = files('sklearn/_build_utils/tempita.py')

py = import('python').find_installation(pure: false)

# Copy all the .py files to the install dir, rather than using
# py.install_sources and needing to list them explicitely one by one
install_subdir('sklearn', install_dir: py.get_install_dir())

subdir('sklearn')

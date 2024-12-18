project(
  'gpuStencil',
  'cpp', 'cuda',
  version: '0.0.1',
  default_options: ['buildtype=release', 'cpp_std=c++20']
)
add_global_arguments('-ccbin=/usr/bin/g++-11', language: 'cuda')

# option('build_examples', type: 'boolean', value: true, description: 'Build the examples')
# option('build_tests', type: 'boolean', value: true, description: 'Build the test examples and run them')

subdir('lib')

if get_option('build_examples')
  subdir('examples')
endif

if get_option('build_tests')
  subdir('tests')
endif

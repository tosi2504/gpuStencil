test1_source = files('1p_1s_1r.cu')
test1 = executable('test1', test1_source, include_directories: lib_include, link_with : [lib_gpuStencil])
test('single point, single site, single rhs', test1)

test2_source = files('1p_1s_multir.cu')
test2 = executable('test2', test2_source, include_directories: lib_include)
test('single point, single site, multi rhs', test2)

test3_source = files('1p_multis_multir.cu')
test3 = executable('test3', test3_source, include_directories: lib_include)
test('single point, multi site, multi rhs', test3)

test4_source = files('multip_multis_multir.cu')
test4 = executable('test4', test4_source, include_directories: lib_include)
test('multi point, multi site, multi rhs', test4)

test5_source = files('multip_multis_multir_GRID.cu')
test5 = executable(
  'test5',
  test5_source,
  include_directories: lib_include,
  link_with: lib_gpuStencil,
  cuda_args : ['--expt-relaxed-constexpr', '--expt-extended-lambda']
)
test('multi point, multi site, multi rhs, GRID', test5)

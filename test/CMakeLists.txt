# Each test is one program that takes a rank count, so the same binary covers
# every world size and ctest reports which size failed.

set(NANOMPI_TESTS
  t_env
  t_pt2pt
  t_coll
  t_reduce
  t_datatype
  t_comm
  t_inplace
  t_team
  t_win)

set(NANOMPI_RANK_COUNTS 1 2 3 4 7 8)

foreach (t IN LISTS NANOMPI_TESTS)
  add_executable(${t} ${t}.c)
  target_link_libraries(${t} PRIVATE nanompi::nanompi)
  if (NOT MSVC)
    target_compile_options(${t} PRIVATE -Wall -Wextra)
    target_link_libraries(${t} PRIVATE m)
  endif()
  foreach (n IN LISTS NANOMPI_RANK_COUNTS)
    add_test(NAME ${t}.n${n} COMMAND ${t} ${n})
    set_tests_properties(${t}.n${n} PROPERTIES TIMEOUT 120)
  endforeach()
endforeach()

# The one test that must NOT create a team: it checks that a plain
# single-threaded caller still sees a working one-rank world.
add_executable(t_solo t_solo.c)
target_link_libraries(t_solo PRIVATE nanompi::nanompi)
if (NOT MSVC)
  target_compile_options(t_solo PRIVATE -Wall -Wextra)
  target_link_libraries(t_solo PRIVATE m)
endif()
add_test(NAME t_solo COMMAND t_solo)
set_tests_properties(t_solo PROPERTIES TIMEOUT 60)

# NANOMPI_NUM_RANKS must be honoured when the program asks for the default.
add_test(NAME t_env.env COMMAND t_env 0)
set_tests_properties(t_env.env PROPERTIES
  ENVIRONMENT "NANOMPI_NUM_RANKS=5"
  PASS_REGULAR_EXPRESSION "ok \\(5 ranks\\)"
  TIMEOUT 60)

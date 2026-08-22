cmake_minimum_required(VERSION 3.16)

project(nanompi
        VERSION 0.1.0
        DESCRIPTION "MPI ranks as threads of one process"
        HOMEPAGE_URL "https://github.com/danielepanozzo/nano-mpi"
        LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NANOMPI_BUILD_TESTS    "Build the nano-mpi test suite" ${PROJECT_IS_TOP_LEVEL})
option(NANOMPI_BUILD_WRAPPERS "Install nanompicc / nanompiexec compiler wrappers" ON)
option(NANOMPI_INSTALL        "Install nano-mpi" ${PROJECT_IS_TOP_LEVEL})

# pthreads on POSIX; on Windows the threading primitives come from the OS
# headers directly (see src/nmpi_port.h) and this resolves to nothing.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
add_library(nanompi src/nanompi.c)
add_library(nanompi::nanompi ALIAS nanompi)

target_include_directories(nanompi
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(nanompi PUBLIC Threads::Threads)

set_target_properties(nanompi PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS ON          # gethostname, sched_yield
  VERSION   ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON
  PUBLIC_HEADER "include/mpi.h;include/nanompi.h"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if (MSVC)
  target_compile_options(nanompi PRIVATE /W4)
  # snprintf, getenv and friends are not "unsafe", they are the standard
  target_compile_definitions(nanompi PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(nanompi PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Compiler wrappers
#
# find_package(MPI) interrogates a wrapper rather than guessing paths, so
# shipping one is what lets an unmodified MPI-consuming project take nano-mpi:
#   cmake -DMPI_C_COMPILER=<prefix>/bin/nanompicc ...
# ---------------------------------------------------------------------------
if (NANOMPI_BUILD_WRAPPERS AND UNIX)
  set(NANOMPI_W_CC   "${CMAKE_C_COMPILER}")
  set(NANOMPI_W_CXX  "${CMAKE_CXX_COMPILER}")
  if (NOT NANOMPI_W_CXX)
    set(NANOMPI_W_CXX "c++")
  endif()
  set(NANOMPI_W_INC  "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
  set(NANOMPI_W_LIB  "${CMAKE_INSTALL_FULL_LIBDIR}")

  foreach (w nanompicc nanompicxx nanompiexec)
    configure_file(cmake/${w}.in ${CMAKE_CURRENT_BINARY_DIR}/${w} @ONLY)
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Install and export
# ---------------------------------------------------------------------------
if (NANOMPI_INSTALL)
  install(TARGETS nanompi
          EXPORT  nanompiTargets
          LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}
          PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(EXPORT nanompiTargets
          FILE      nanompiTargets.cmake
          NAMESPACE nanompi::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanompi)

  # The build-tree half of the same export. A parent project that adds nano-mpi
  # as a subdirectory and then exports its own targets needs this one to exist:
  # CMake refuses to export a target that links a target which is in no export
  # set at all, and hypre does exactly that.
  export(EXPORT nanompiTargets
         FILE      "${CMAKE_CURRENT_BINARY_DIR}/nanompiTargets.cmake"
         NAMESPACE nanompi::)

  configure_package_config_file(
    cmake/nanompi-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/nanompi-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanompi)

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/nanompi-config-version.cmake
    COMPATIBILITY SameMajorVersion)

  install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/nanompi-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/nanompi-config-version.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nanompi)

  if (NANOMPI_BUILD_WRAPPERS AND UNIX)
    install(PROGRAMS
              ${CMAKE_CURRENT_BINARY_DIR}/nanompicc
              ${CMAKE_CURRENT_BINARY_DIR}/nanompicxx
              ${CMAKE_CURRENT_BINARY_DIR}/nanompiexec
            DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  install(FILES SCOPE.md README.md LICENSE
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/nanompi)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if (NANOMPI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/util/forall.hpp"

/*
 *  Possible CHAI/RAJA bug
 *
 * Copy constructor is called multiple times, we don't know why.
 */

#if defined(RAJA_ENABLE_CUDA)

class FOO
{
public:
  FOO():m_chai()
  {}

  __host__ __device__ FOO( const FOO & source ):m_chai()
  {
#if !defined(__CUDA_ARCH__)
    printf("calling FOO copy constructor\n");
#endif
  }

  double m_chai[10];
};



class CHAIWRAPPER
{
public:
  CHAIWRAPPER():m_chai(10)
  {}

  __host__ __device__ CHAIWRAPPER( const CHAIWRAPPER & source ):m_chai(source.m_chai)
  {
#if !defined(__CUDA_ARCH__)
    printf("calling CHAIWRAPPER copy constructor\n");
#endif
  }

  chai::ManagedArray<double> m_chai;
};
#endif


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

#if defined(RAJA_ENABLE_CUDA)
  chai::ManagedArray<double> chaiArray(10);
  FOO foo;
  CHAIWRAPPER wrapper;
  
  printf("executing kernel... \n", i);

  RAJA::forall<RAJA::cuda_exec<10>>(RAJA::RangeSegment(0,10), [=] __device__ (int i) {
      printf("i = %d \n", i);
      chaiArray[i] = i;
      foo.m_chai[i] = i;
      wrapper.m_chai[i] = i;
    });
  printf("done executing kernel! \n", i);
#endif


  return 0;
}

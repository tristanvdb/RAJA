/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_sequential_HXX
#define RAJA_scan_sequential_HXX

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read raja/README-license.txt.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hxx"

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>

namespace RAJA
{
namespace detail
{
namespace scan
{

template <typename Iter, typename BinFn, typename T>
void inclusive_inplace(const ::RAJA::seq_exec&,
                       Iter begin,
                       Iter end,
                       BinFn f,
                       T v)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  Value agg = *begin;
  while (++begin != end) {
    agg = f(*begin, agg);
    *begin = agg;
  }
}

template <typename Iter>
void inclusive_inplace(const ::RAJA::seq_exec& exec, Iter begin, Iter end)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  inclusive_inplace(exec, begin, end, std::plus<Value>{}, Value{0});
}

template <typename Iter, typename BinFn, typename T>
void exclusive_inplace(const ::RAJA::seq_exec&,
                       Iter begin,
                       Iter end,
                       BinFn f,
                       T v)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  const int n = end - begin;
  Value agg = v;
  for (int i = 0; i < n; ++i) {
    Value t = *(begin + i);
    *(begin + i) = agg;
    agg = f(agg, t);
  }
}

template <typename Iter>
void exclusive_inplace(const ::RAJA::seq_exec& exec, Iter begin, Iter end)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  exclusive_inplace(exec, begin, end, std::plus<Value>{}, Value{0});
}

template <typename Iter, typename OutIter, typename BinFn, typename T>
void inclusive(const ::RAJA::seq_exec&,
               Iter begin,
               Iter end,
               OutIter out,
               BinFn f,
               T v)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  Value agg = *begin;
  *out++ = agg;
  for (Iter i = begin + 1; i != end; ++i) {
    agg = f(agg, *i);
    *out++ = agg;
  }
}

template <typename Iter, typename OutIter>
void inclusive(const ::RAJA::seq_exec& exec, Iter begin, Iter end, OutIter out)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  inclusive(exec, begin, end, out, std::plus<Value>{}, Value{0});
}

template <typename Iter, typename OutIter, typename BinFn, typename T>
void exclusive(const ::RAJA::seq_exec&,
               Iter begin,
               Iter end,
               OutIter out,
               BinFn f,
               T v)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  Value agg = v;
  OutIter o = out;
  *o++ = v;
  for (Iter i = begin; i != end - 1; ++i, ++o) {
    agg = f(*i, agg);
    *o = agg;
  }
}

template <typename Iter, typename OutIter>
void exclusive(const ::RAJA::seq_exec& exec, Iter begin, Iter end, OutIter out)
{
  using Value = typename std::iterator_traits<Iter>::value_type;
  exclusive(exec, begin, end, out, std::plus<Value>{}, Value{0});
}

}  // namespace scan

}  // namespace detail

}  // namespace RAJA

#endif

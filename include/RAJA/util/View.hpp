/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a multi-dimensional view class.
 *
 ******************************************************************************
 */

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

#ifndef RAJA_VIEW_HPP
#define RAJA_VIEW_HPP

#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/pattern/atomic.hpp"

#include "RAJA/util/Layout.hpp"

#if defined(RAJA_ENABLE_CHAI)
#include "chai/ManagedArray.hpp"
#endif

namespace RAJA
{

template <
  typename ValueType,
  typename PointerType,
  typename ReferenceType,
  typename NonConstPointerType = typename std::add_pointer<
                                   typename std::remove_const<
                                     typename std::remove_pointer<PointerType>::type
                                   >::type
                                 >::type
>
struct DefaultViewConfigHelper {
  using reference_type = ReferenceType;

  using value_type = ValueType;
  using pointer_type = PointerType;

  using nc_value_type = typename std::remove_const<value_type>::type;
  using nc_pointer_type = NonConstPointerType;

  template <typename LayoutT, typename... Args>
  static inline reference_type get_reference(PointerType const data, LayoutT const & layout, Args... args) {
    auto idx = stripIndexType(layout(args...));
    reference_type value = data[idx];
    return value;
  }
};

#if defined(RAJA_ENABLE_ZFP)

#define DEBUG_ZFP_WITH_PRINTF 0

struct view_config_user_defined_array          {}; // must define: type `array_type` which must define `reference_t`
struct view_config_array_of_user_defined_array {}; // must define: type `array_type` which must define `reference_t`
                                                   //              field `user_array_on_fast_dims` of type bool   (static constexpr)
                                                   //              field `num_slow_dims` of type size_t (static constexpr)

template<typename T, typename = void>
struct has_type : std::false_type { };

template<typename T>
struct has_type<T, decltype(sizeof(typename T::type), void())> : std::true_type { };

template <
  typename ValueType,
  typename PointerType,
  bool is_user_defined_array          = std::is_base_of<view_config_user_defined_array,          PointerType>::value,
  bool is_array_of_user_defined_array = std::is_base_of<view_config_array_of_user_defined_array, PointerType>::value
>
struct ViewConfigHelper;

template < typename ValueType, typename PointerType>
struct ViewConfigHelper<ValueType, PointerType, true, true> {
  static_assert(std::is_same<PointerType, int>::value, "Does not make sense!!!");
};

template < typename ValueType, typename PointerType >
struct ViewConfigHelper<ValueType, PointerType, false, false>
       : DefaultViewConfigHelper <ValueType, PointerType, ValueType & >
{};

template < typename ValueType, typename ConfigType >
struct ViewConfigHelper<ValueType, ConfigType, true, false>
       : DefaultViewConfigHelper <ValueType, typename ConfigType::array_type::pointer, typename ConfigType::array_type::reference , typename std::remove_const< typename ConfigType::array_type >::type::pointer >
{};


template <typename PointerType, typename LayoutT, size_t n_dims, size_t slow_dims, bool user_array_on_fast_dims>
struct SizeHelper {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    static_assert(n_dims > 1);
    static_assert(slow_dims < n_dims);

    for (size_t i = 0; i < n_dims; i++) {
      if (i < slow_dims) {
        slow_size *= layout.sizes[i];
      } else {
        fast_size *= layout.sizes[i];
      }
    }
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 2, 1, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0];
    fast_size = layout.sizes[1];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 3, 1, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0];
    fast_size = layout.sizes[1] * layout.sizes[2];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 3, 2, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0] * layout.sizes[1];
    fast_size = layout.sizes[2];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 4, 1, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0];
    fast_size = layout.sizes[1] * layout.sizes[2] * layout.sizes[3];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 4, 2, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0] * layout.sizes[1];
    fast_size = layout.sizes[2] * layout.sizes[3];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 4, 3, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0] * layout.sizes[1] * layout.sizes[2];
    fast_size = layout.sizes[3];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 5, 1, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0];
    fast_size = layout.sizes[1] * layout.sizes[2] * layout.sizes[3] * layout.sizes[4];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 5, 2, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0] * layout.sizes[1];
    fast_size = layout.sizes[2] * layout.sizes[3] * layout.sizes[4];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 5, 3, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0] * layout.sizes[1] * layout.sizes[2];
    fast_size = layout.sizes[3] * layout.sizes[4];
  }
};

template <typename PointerType, typename LayoutT, bool user_array_on_fast_dims>
struct SizeHelper<PointerType, LayoutT, 5, 4, user_array_on_fast_dims> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    slow_size = layout.sizes[0] * layout.sizes[1] * layout.sizes[2] * layout.sizes[3];
    fast_size = layout.sizes[4];
  }
};

template <typename PointerType, typename LayoutT, size_t slow_dims>
struct SizeHelper<PointerType, LayoutT, 1, slow_dims, false> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    size_t full_size = layout.sizes[0];
    slow_size = data[0].size();
    fast_size = full_size / slow_size;
  }
};

template <typename PointerType, typename LayoutT, size_t slow_dims>
struct SizeHelper<PointerType, LayoutT, 1, slow_dims, true> {
  inline static void get_sizes(PointerType const data, LayoutT const & layout, size_t & slow_size, size_t & fast_size) {
    size_t full_size = layout.sizes[0];
    fast_size = data[0].size();
    slow_size = full_size / fast_size;
  }
};  

template < typename ValueType, typename ConfigType >
struct ViewConfigHelper<ValueType, ConfigType, false, true>
{
  using array_type = typename ConfigType::array_type;
  using reference_type = typename array_type::reference;

  using value_type = ValueType;
  using pointer_type = typename array_type::pointer *;

  using nc_value_type = typename std::remove_const<value_type>::type;
  using nc_pointer_type = typename std::remove_const< typename ConfigType::array_type >::type *;

  template <typename LayoutT, typename... Args>
  inline static reference_type get_reference(pointer_type const data, LayoutT const & layout, Args... args) {
    auto idx = stripIndexType(layout(args...));
#if DEBUG_ZFP_WITH_PRINTF
    printf("ViewConfigHelper::get_reference(...):\n");
    printf(" -- idx = %d\n", idx);
    printf(" -- LayoutT::n_dims = %d\n", LayoutT::n_dims);
#endif
    size_t slow_size = 1;
    size_t fast_size = 1;
    SizeHelper<
      pointer_type, LayoutT, LayoutT::n_dims, ConfigType::num_slow_dims, ConfigType::user_array_on_fast_dims
    >::get_sizes(
      data, layout, slow_size, fast_size
    );
#if DEBUG_ZFP_WITH_PRINTF
    printf(" -- slow_size = %zu\n", slow_size);
    printf(" -- fast_size = %zu\n", fast_size);
#endif
    size_t main_index = 0;
    size_t user_index = 0;
    if (ConfigType::user_array_on_fast_dims) {
      main_index = idx / fast_size;
      user_index = idx % fast_size;
    } else {
      main_index = idx / slow_size;
      user_index = idx % slow_size;
    }
#if DEBUG_ZFP_WITH_PRINTF
    printf(" -- main_index = %zu\n", main_index);
    printf(" -- user_index = %zu\n", user_index);
#endif
    return data[main_index][user_index];
  }
};

#else
template <typename ValueType, typename PointerType>
using ViewConfigHelper = DefaultViewConfigHelper <ValueType, PointerType, ValueType & >;
#endif

template <typename ValueType,
          typename LayoutType,
          typename PointerType = ValueType *>
struct View {

  using ViewConfig = ViewConfigHelper<ValueType, PointerType>;

  using reference_type = typename ViewConfig::reference_type;

  using value_type = typename ViewConfig::value_type;
  using pointer_type = typename ViewConfig::pointer_type;

  using nc_value_type = typename ViewConfig::nc_value_type;
  using nc_pointer_type = typename ViewConfig::nc_pointer_type;

  using layout_type = LayoutType;
  using NonConstView = View<nc_value_type, layout_type, nc_pointer_type>;

//static_assert(std::is_same<layout_type, int>::value);

  layout_type const layout;
  pointer_type data;

  template <typename... Args>
  RAJA_INLINE constexpr View(pointer_type data_ptr, Args... dim_sizes)
      : layout(dim_sizes...), data(data_ptr)
  {
  }

  RAJA_INLINE constexpr View(pointer_type data_ptr, layout_type &&layout)
      : layout(layout), data(data_ptr)
  {
  }

  //We found the compiler-generated copy constructor does not actually copy-construct
  //the object on the device in certain nvcc versions. 
  //By explicitly defining the copy constructor we are able ensure proper behavior.
  //Git-hub pull request link https://github.com/LLNL/RAJA/pull/477
  RAJA_INLINE RAJA_HOST_DEVICE constexpr View(View const &V)
      : layout(V.layout), data(V.data)
  {
  }

  template <bool IsConstView = std::is_const<value_type>::value>
  RAJA_INLINE constexpr View(
          typename std::enable_if<IsConstView, NonConstView>::type const &rhs)
      : layout(rhs.layout), data(rhs.data)
  {
  }

  RAJA_INLINE void set_data(pointer_type data_ptr) { data = data_ptr; }

  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template <typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE reference_type operator()(Args... args) const
  {
    return ViewConfig::get_reference(data, layout, args...);
  }
};

template <typename ValueType,
          typename PointerType,
          typename LayoutType,
          typename... IndexTypes>
struct TypedViewBase {
  using Base = View<ValueType, LayoutType, PointerType>;

  Base base_;

  template <typename... Args>
  RAJA_INLINE constexpr TypedViewBase(typename Base::pointer_type data_ptr, Args... dim_sizes)
      : base_(data_ptr, dim_sizes...)
  {
  }

  template <typename CLayoutType>
  RAJA_INLINE constexpr TypedViewBase(typename Base::pointer_type data_ptr,
                                      CLayoutType &&layout)
      : base_(data_ptr, std::forward<CLayoutType>(layout))
  {
  }

  RAJA_INLINE void set_data(typename Base::pointer_type data_ptr) { base_.set_data(data_ptr); }

  RAJA_HOST_DEVICE RAJA_INLINE typename Base::value_type &operator()(IndexTypes... args) const
  {
    return base_.operator()(stripIndexType(args)...);
  }
};

template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedView =
    TypedViewBase<ValueType, ValueType *, LayoutType, IndexTypes...>;

#if defined(RAJA_ENABLE_CHAI)

template <typename ValueType, typename LayoutType>
using ManagedArrayView =
    View<ValueType, LayoutType, chai::ManagedArray<ValueType>>;


template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedManagedArrayView = TypedViewBase<ValueType,
                                            chai::ManagedArray<ValueType>,
                                            LayoutType,
                                            IndexTypes...>;

#endif

template <typename ViewType, typename AtomicPolicy = RAJA::atomic::auto_atomic>
struct AtomicViewWrapper {
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using atomic_type = RAJA::atomic::AtomicRef<value_type, AtomicPolicy>;

  base_type base_;

  RAJA_INLINE
  constexpr explicit AtomicViewWrapper(ViewType view) : base_(view) {}

  RAJA_INLINE void set_data(pointer_type data_ptr) { base_.set_data(data_ptr); }

  template <typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE atomic_type operator()(ARGS &&... args) const
  {
    return atomic_type(&base_.operator()(std::forward<ARGS>(args)...));
  }
};


/*
 * Specialized AtomicViewWrapper for seq_atomic that acts as pass-thru
 * for performance
 */
template <typename ViewType>
struct AtomicViewWrapper<ViewType, RAJA::atomic::seq_atomic> {
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using atomic_type =
      RAJA::atomic::AtomicRef<value_type, RAJA::atomic::seq_atomic>;

  base_type base_;

  RAJA_INLINE
  constexpr explicit AtomicViewWrapper(ViewType const &view) : base_{view} {}

  RAJA_INLINE void set_data(pointer_type data_ptr) { base_.set_data(data_ptr); }

  template <typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE value_type &operator()(ARGS &&... args) const
  {
    return base_.operator()(std::forward<ARGS>(args)...);
  }
};


template <typename AtomicPolicy, typename ViewType>
RAJA_INLINE AtomicViewWrapper<ViewType, AtomicPolicy> make_atomic_view(
    ViewType const &view)
{

  return RAJA::AtomicViewWrapper<ViewType, AtomicPolicy>(view);
}


}  // namespace RAJA

#endif

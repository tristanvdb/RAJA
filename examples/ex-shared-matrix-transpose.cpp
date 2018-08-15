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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Shared Memory Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N x N is
 *  reconfigured as a second matrix At with the rows of 
 *  matrix A reorganized as the columns of At and the columns
 *  of matrix A becoming be the rows of matrix At. 
 *
 *  This operation is carried out using a shared memory tiling 
 *  algorithm. The algorithm first loads matrix entries into a 
 *  thread shared tile, a small two-dimensional array, and then 
 *  reads from the tile swapping the row and column indices for 
 *  the output matrix.
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loop will load/read
 *  data into the tile; while outer loops will iterate over the number
 *  of tiles needed to carry out the transposition. For simplicity we assume
 *  the tile size divides the number of rows and columns of the matrix.
 *
 *  RAJA variants of the example construct a tile object using a RAJA shared memory
 *  window. For CPU execution, RAJA shared memory windows can be used to improve
 *  performance via cache blocking. For CUDA GPU execution, RAJA shared memory
 *  is mapped to CUDA shared memory.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *       - Multiple lambdas
 *       - Shared memory tiling windows
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//#define OMP_EX_1 //Breaks kernel

//
// Define dimensionality of matrices
//
const int DIM = 2;

struct data_s{
  int array[2][3];

  int& operator()(RAJA::Index_type row, RAJA::Index_type col){
    return array[row][col];
  }
  
};

//
// Function for checking results
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Aview,
                 RAJA::View<T, RAJA::Layout<DIM>> Atview,
                 int Nrows, int Ncols);

//
// Function for printing results
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int Nrows, int Ncols);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA shared matrix transpose example...\n";

  //
  // Define num rows/cols in matrix
  //
  const int factor = 2;
  const int Nrows  = 2*factor;
  const int Ncols  = 3*factor;

  //
  // Allocate matrix data
  //
  int *A  = memoryManager::allocate<int>(Nrows * Ncols);
  int *At = memoryManager::allocate<int>(Nrows * Ncols);

  //
  // In the following implementations of shared matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, Nrows, Ncols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, Ncols, Nrows); //transpose

  //
  // Define TILE dimensions
  //
  const int TILE_DIM1   = 2;
  const int TILE_DIM0   = 3;

  //
  // Define bounds for inner and outer loops
  //
  const int inner_Dim1 = TILE_DIM1;
  const int inner_Dim0 = TILE_DIM0;

  const int outer_Dim1 = Nrows/TILE_DIM1;
  const int outer_Dim0 = Ncols/TILE_DIM0;

  //
  // Initialize matrix data
  //
  for (int row = 0; row < Nrows; ++row) {
    for (int col = 0; col < Ncols; ++col) {
      Aview(row, col) = col + Ncols * row;
    }
  }

  printf("Printing intial matrix ... \n");
  printResult<int>(Aview, Nrows, Ncols);
  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, Nrows * Ncols * sizeof(int));
  
  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dim1; ++by) {
    for (int bx = 0; bx < outer_Dim0; ++bx) {

      //TILE 2 x 3 
      int TILE[inner_Dim1][inner_Dim0];

      for (int ty = 0; ty < inner_Dim1; ++ty) { // 0 - 1
        for (int tx = 0; tx < inner_Dim0; ++tx) {//0 - 3
          int col = bx * inner_Dim0 + tx;  // Matrix column index
          int row = by * inner_Dim1 + ty;  // Matrix row index
          TILE[ty][tx] = Aview(row, col);
        }
      }

      for (int tx = 0; tx < inner_Dim0; ++tx) { //0 - 3
        for (int ty = 0; ty < inner_Dim1; ++ty) { // 0 - 1
          int col = bx * inner_Dim0 + tx;  // Transposed matrix column index
          int row = by * inner_Dim1 + ty;  // Transposed matrix row index
          Atview(col, row) = TILE[ty][tx];
        }
      }

    }
  }
 
  //----------------------------------------------------------------------------//

  //
  // The following RAJA variants use the RAJA::kernel method to carryout the
  // transpose.
  //
  // Here, we define RAJA range segments to establish the iteration spaces for
  // the inner and outer loops.
  //
  RAJA::RangeSegment inner_Range0(0, inner_Dim0);
  RAJA::RangeSegment inner_Range1(0, inner_Dim1);
  RAJA::RangeSegment outer_Range0(0, outer_Dim0);
  RAJA::RangeSegment outer_Range1(0, outer_Dim1);

  //
  // Iteration spaces are stored within a RAJA tuple
  //
  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0,Ncols), RAJA::RangeSegment(0,Nrows));

  //----------------------------------------------------------------------------//
  std::cout << "\n Running RAJA shared matrix transpose ...\n";
  std::memset(At, 0, Nrows * Ncols * sizeof(int));

  //
  // Next, we construct a shared memory window object
  // to represent the tile load/read matrix entries.
  // The shared memory object constructor is templated on:
  // 1) RAJA shared memory type
  // 2) Data type
  // 3) List of lambda arguments which will be accessing the data
  // 4) Dimension of the objects
  // 5) The type of the tuple holding the iterations spaces
  //    (for simplicity decltype is used to infer the object type)
  //

  printf("constructing a shared memory type \n");
  using seq_shmem_t = RAJA::ShmemTile<RAJA::omp_shmem,
                                      int,
                                      RAJA::ArgList<1, 0>,
                                      RAJA::SizeList<TILE_DIM1, TILE_DIM0>,
                                      decltype(iSpace)>; //picks up the size of segments


  seq_shmem_t RAJA_SEQ_TILE;
  printf("instantiated an object \n");

  
  using KERNEL_EXEC_POL = 
    RAJA::KernelPolicy<
       RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM1>, RAJA::loop_exec,
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM0>, RAJA::loop_exec,
                            //RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM0>, RAJA::loop_exec

         RAJA::statement::SetShmemWindow<
           RAJA::statement::For<1, RAJA::loop_exec,
             RAJA::statement::For<0, RAJA::loop_exec,
               RAJA::statement::Lambda<0>
              > //closes For 0
            > //closes For 1
           
           //,RAJA::statement::OmpSyncThreads
           
           ,RAJA::statement::For<0, RAJA::loop_exec, 
             RAJA::statement::For<1, RAJA::loop_exec,
               RAJA::statement::Lambda<1>
            > //closes For 0
           > //closes For 1

           > // closes shmem window
          > // closes Tile 0
        > // closes Tile 1
    >; // closes policy list

#if 1
  RAJA::kernel_param<KERNEL_EXEC_POL>(

      iSpace,                                      
                                      
      //RAJA::tuple<double[2]>({0,0}),
      RAJA::tuple<data_s>(data_s{}),
      
      //RAJA::make_tuple(RAJA_SEQ_TILE),
      //
      // (1) Lambda for inner loops to load data into the tile
      //
      //[=](RAJA::Index_type col, RAJA::Index_type row, seq_shmem_t &RAJA_SEQ_TILE, double& dot) {
      [=](RAJA::Index_type col, RAJA::Index_type row, data_s& dot) {

        int tx = col % TILE_DIM0;
        int ty = row % TILE_DIM1;

        dot(ty, tx) = Aview(row, col);
      }
      ,      
      [=](RAJA::Index_type col, RAJA::Index_type row, data_s& dot) {

        int tx = col % TILE_DIM0;
        int ty = row % TILE_DIM1;

        Atview(col, row) = dot(ty, tx);
      }

     );
#endif
  checkResult<int>(Aview, Atview, Nrows, Ncols);
  printResult<int>(Atview, Ncols, Nrows);

  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(At);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Aview,
                 RAJA::View<T, RAJA::Layout<DIM>> Atview,
                 int Nrows, int Ncols)
{
  bool match = true;
  for (int row = 0; row < Nrows; ++row) {
    for (int col = 0; col < Ncols; ++col) {
      if (Aview(row, col) != Atview(col, row) ) {
        match = false;
      }
    }
  }
  if (match) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Function to print result.
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int Nrows, int Ncols)
{
  std::cout << std::endl;
  for (int row = 0; row < Nrows; ++row) {
    for (int col = 0; col < Ncols; ++col) {
      std::cout << Atview(row, col)<<" ";
    }
    std::cout<<"\n";
  }
  std::cout << std::endl;
}

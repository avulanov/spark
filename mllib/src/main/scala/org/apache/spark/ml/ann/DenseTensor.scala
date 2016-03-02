/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.ann

import breeze.linalg.{DenseVector, DenseMatrix}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

/**
 * Dense tensor column-major representation. // TODO: row major??
 * @param data underlying data
 * @param tensorShape shape of tensor
 * @param offset offset in the data
 * @tparam T type
 */
class DenseTensor[@specialized(Double, Float) T] private[ann] (
  val data: Array[T],
  tensorShape: Array[Int],
  val offset: Int) {
  private val actualSize = data.length - offset
  private var _isTransposed = false
  val requiredSize = tensorShape.product
  require(requiredSize <= actualSize,
    "Actual size of the array does not correspond to dimension Sizes")
  private var myShape = tensorShape

  /**
   * Tensor size (tensor data array might be bigger)
   * @return
   */
  def size: Int = {
    myShape.product
  }

  /**
   * Shape of the tensor
   * @return shape
   */
  def shape: Array[Int] = myShape

  /**
   * Reshape the tensor. Supports reshaping within the same data size
   * @param newShape new shape
   * @return reshaped tensor backed by the same data
   */
  def reshape(newShape: Array[Int]): DenseTensor[T] = {
    val requiredSize = newShape.product
    require(requiredSize == actualSize)
    myShape = newShape
    this
  }

  /**
   * Get the value at position index
   * @param index index
   * @return value
   */
  def value(index: Array[Int]): T = {
    data(offset(index))
  }

  private def offset(index: Array[Int]): Int = {
    var offset = index.last
    for (i <- myShape.length - 1 to 1 by -1) {
      offset = index(i - 1) + myShape(i - 1) * offset
    }
    offset
  }

  /**
   * Check if tensor is transposed
   * @return true if transposed, false otherwise
   */
  def transposed: Boolean = _isTransposed

  /**
   * Transpose tensor. Does not actually transpose the data.
   * It is used for operations such as gemm.
   * @return self
   */
  def transpose: DenseTensor[T] = {
    _isTransposed = true
    this
  }

  /**
   * Slice the tensor by the last dimension
   * @param from index
   * @param until index
   * @return tensor backed by the same data
   */
  def slice(from: Int, until: Int): DenseTensor[T] = {
    require(from < until && from < myShape(0) && until <= myShape(0),
      "start and end must be within the size of first dimension, also start <= end")
    val shapeInit = myShape.init
    val lastDimensionNewSize = until - from
    val startOffset = offset(shapeInit.map(_ => 0) :+ from)
    new DenseTensor[T](data, shapeInit :+ lastDimensionNewSize, startOffset)
  }

  /**
   * Slice the tensor by one index in the last dimension
   * @param index index
   * @return squeezed tensor
   */
  def slice(index: Int): DenseTensor[T] = {
    slice(index, index + 1).squeeze()
  }

  /**
   * Squeze the dimensions of size 1
   * @return tensor backed by the same data
   */
  def squeeze(): DenseTensor[T] = {
    val buf = new ArrayBuffer[Int](myShape.length)
    for (dim <- myShape) {
      if (dim > 1) buf += dim
    }
    myShape = buf.toArray
    this
  }

  /**
   * Copy the underlying data
   * @param m ClassTag
   * @return data array
   */
  def copyData()(implicit m: ClassTag[T]): Array[T] = {
    val array = new Array[T](myShape.product)
    System.arraycopy(data, offset, array, 0, array.length)
    array
  }

  /**
   * Fill tensor with the data from the other tensor
   * @param donor tensor from which to get data
   * @return self
   */
  def fillWith(donor: DenseTensor[T]): DenseTensor[T] = {
    require(size % donor.size == 0 && size >= donor.size,
      "data size of recipient tensor must be >= and divide evenly by the data size of donor tensor")
    val donorSize = donor.size
    val numCopies = size / donorSize
    var k = 0
    var nextOffset = 0
    while (k <  numCopies) {
      System.arraycopy(donor.data, donor.offset, this.data, this.offset + nextOffset, donorSize)
      nextOffset += donorSize
      k += 1
    }
    this
  }
}

object DenseTensor {

  /**
   * Create a tensor with zeros
   * @param tensorShape shape
   * @param m ClassTag
   * @tparam T implicit type
   * @return tensor
   */
  def apply[@specialized(Double, Float) T](tensorShape: Array[Int])
                                          (implicit m: ClassTag[T]): DenseTensor[T] = {
    val data: Array[T] = new Array[T](tensorShape.product)
    DenseTensor(data, tensorShape)
  }

  /**
   * Create a tensor from data
   * @param data data
   * @param tensorShape shape
   * @param offset offset in the data
   * @param m ClassTag
   * @tparam T implicit type
   * @return tensor
   */
  def apply[T](data: Array[T], tensorShape: Array[Int], offset: Int = 0)
              (implicit m: ClassTag[T]): DenseTensor[T] = {
    new DenseTensor[T](data, tensorShape, offset)
  }

  /**
   * Create and fill tensor with values
   * @param tensorShape shape
   * @param elem value
   * @param m ClassTag
   * @tparam T type
   * @return tensor
   */
  def fill[@specialized(Double, Float) T](tensorShape: Array[Int])
                                         (elem: => T)
                                         (implicit m: ClassTag[T]): DenseTensor[T] = {
    val data: Array[T] = Array.fill[T](tensorShape.product)(elem)
    DenseTensor(data, tensorShape)
  }

  /**
   * Apply a function to tensor x and put the result in the y
   * @param x source
   * @param y result
   * @param func function
   * @tparam T type
   */
  def applyFunction[T](x: DenseTensor[T], y: DenseTensor[T], func: T => T): Unit = {
    require(x.size == y.size, "Tensor sizes must be equal")
    var i = y.offset
    while (i < y.size) {
      y.data(i) = func(x.data(i))
      i += 1
    }
  }

  /**
   * Apply a function to tensor x and put the result in the y
   * @param x1 source1
   * @param x2 source2
   * @param y result
   * @param func function
   * @tparam T type
   */
  def applyFunction[T](
  x1: DenseTensor[T],
  x2: DenseTensor[T],
  y: DenseTensor[T],
  func: (T, T) => T): Unit = {
    require(x1.size == y.size && x2.size == y.size, "Tensor sizes must be equal")
    var i = y.offset
    while (i < y.size) {
      y.data(i) = func(x1.data(i), x2.data(i))
      i += 1
    }
  }

  /**
   * Double 2d tensor multiplication C <- alpha * A * B + beta * C
   * @param alpha alpha
   * @param a A
   * @param b B
   * @param beta beta
   * @param c C
   */
  def gemm(
  alpha: Double,
  a: DenseTensor[Double],
  b: DenseTensor[Double],
  beta: Double,
  c: DenseTensor[Double]): Unit = {
    // TODO: case with 3d and more
    require(a.shape.length == 2 && b.shape.length == 2 && c.shape.length == 2,
      "A, B, or C are not 2d tensors")
    // TODO: add code if matrices isTranspose!!!
    require(a.shape(1) == b.shape(0), "A & B Dimension mismatch!")
    require(a.shape(0) == c.shape(0), "A & C Dimension mismatch!")
    require(b.shape(1) == c.shape(1), "B & C Dimension mismatch!")
    NativeBLAS.dgemm(transposeString(a), transposeString(b), c.shape(0), c.shape(1), a.shape(1),
    // TODO: check majorStride
      alpha, a.data, a.offset, a.shape(0) /* a.majorStride */,
      b.data, b.offset, b.shape(0) /* b.majorStride */,
      beta, c.data, c.offset, c.shape(0))
  }

  /**
    * Double 2d tensor multiplication C <- alpha * A * B + beta * C
    * @param alpha alpha
    * @param a A
    * @param b B
    * @param beta beta
    * @param c C
    */
  def gemm(
             alpha: Float,
             a: DenseTensor[Float],
             b: DenseTensor[Float],
             beta: Float,
             c: DenseTensor[Float]): Unit = {
    // TODO: case with 3d and more
    require(a.shape.length == 2 && b.shape.length == 2 && c.shape.length == 2,
      "A, B, or C are not 2d tensors")
    // TODO: add code if matrices isTranspose!!!
    require(a.shape(1) == b.shape(0), "A & B Dimension mismatch!")
    require(a.shape(0) == c.shape(0), "A & C Dimension mismatch!")
    require(b.shape(1) == c.shape(1), "B & C Dimension mismatch!")
    NativeBLAS.sgemm(transposeString(a), transposeString(b), c.shape(0), c.shape(1), a.shape(1),
      // TODO: check majorStride
      alpha, a.data, a.offset, a.shape(0) /* a.majorStride */,
      b.data, b.offset, b.shape(0) /* b.majorStride */,
      beta, c.data, c.offset, c.shape(0))
  }

  private def transposeString[T](a: DenseTensor[T]): String = if (a.transposed) "T" else "N"

  /**
   * GEMV: y := alpha * A * x + beta * y
   * @param alpha alpha
   * @param a A
   * @param x x
   * @param beta beta
   * @param y y
   */
  def gemv(
  alpha: Double,
  a: DenseTensor[Double],
  x: DenseTensor[Double],
  beta: Double,
  y: DenseTensor[Double]): Unit = {
    require(a.shape.length == 2 && x.shape.length == 1 && y.shape.length == 1,
      "A must be 2d and X, Y - 1d tensors")
    require(a.shape(1) == x.shape(0), "A & X Dimension mismatch!")
    require(a.shape(0) == y.shape(0), "A & Y Dimension mismatch!")
    NativeBLAS.dgemv(transposeString(a), a.shape(0), a.shape(1),
      alpha, a.data, a.offset, a.shape(0) /* a.majorStride */,
      x.data, x.offset, 1 /* x.shape(0) */ /* x.stride */,
      beta, y.data, y.offset, 1 /* y.shape(0) */ /* y.stride */)
  }

  /**
   * GEMV: y := alpha * A * x + beta * y
   * @param alpha alpha
   * @param a A
   * @param x x
   * @param beta beta
   * @param y y
   */
  def gemv(
  alpha: Float,
  a: DenseTensor[Float],
  x: DenseTensor[Float],
  beta: Float,
  y: DenseTensor[Float]): Unit = {
    require(a.shape.length == 2 && x.shape.length == 1 && y.shape.length == 1,
      "A must be 2d and X, Y - 1d tensors")
    require(a.shape(1) == x.shape(0), "A & X Dimension mismatch!")
    require(a.shape(0) == y.shape(0), "A & Y Dimension mismatch!")
    NativeBLAS.sgemv(transposeString(a), a.shape(0), a.shape(1),
      alpha, a.data, a.offset, a.shape(0) /* a.majorStride */,
      x.data, x.offset, 1 /* x.shape(0) */ /* x.stride */,
      beta, y.data, y.offset, 1 /* y.shape(0) */ /* y.stride */)
  }

  protected def elementwise(
  a: DenseTensor[Double],
  b: DenseTensor[Double],
  op: (Double, Double) => Double): Unit = {
    require(a.size == b.size, "Tensors of different size")
    var i = 0
    while (i < a.size) {
      a.data(i) = op(a.data(i), b.data(i))
      i += 1
    }
  }

  def elementwiseProduct(a: DenseTensor[Double], b: DenseTensor[Double]): Unit = {
    elementwise(a, b, (x, y) => x * y)
  }
}

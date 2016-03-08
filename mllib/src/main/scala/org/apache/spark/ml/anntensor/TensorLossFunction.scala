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

package org.apache.spark.ml.anntensor

import java.util.Random

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum => Bsum}
import breeze.numerics.{log => Blog}
import org.apache.spark.ml.ann.DenseTensor
import org.apache.spark.ml.anntensor

import AnnTypes._
/**
  * Trait for loss function
  */
private[anntensor] trait LossFunction {
  /**
    * Loss function
    *
    * @param output actual output
    * @param target target output
    * @param delta output delta to write to
    * @return
    */
  def loss(output: Tensor, target: Tensor, delta: Tensor): Double
}

private[ml] class SigmoidLayerWithSquaredError extends anntensor.Layer {
  override val weightSize = 0
  override def outputSize(inputSize: Int): Int = inputSize
  override val inPlace = true
  override def model(weights: Tensor): anntensor.LayerModel = new SigmoidLayerModelWithSquaredError()
  override def initModel(weights: Tensor, random: Random): anntensor.LayerModel =
    new SigmoidLayerModelWithSquaredError()
}

private[anntensor] class SigmoidLayerModelWithSquaredError
  extends anntensor.FunctionalLayerModel(new anntensor.FunctionalLayer(new anntensor.SigmoidFunction)) with LossFunction {
  override def loss(output: Tensor, target: Tensor, delta: Tensor): Double = {
    //anntensor.UniversalFunction(output, target, delta, (o: Double, t: Double) => o - t)
//    val error = Bsum(delta :* delta) / 2 / output.cols
//    anntensor.UniversalFunction(delta, output, delta, (x: Double, o: Double) => x * (o - o * o))
    DenseTensor.applyFunction(output, target, delta, (o: Double, t: Double) => o - t)
    // TODO: implement
    val error = 0 // Bsum(delta :* delta) / 2 / output.cols
    DenseTensor.applyFunction(delta, output, delta, (x: Double, o: Double) => x * (o - o * o))
    error
  }
}

private[ml] class SoftmaxLayerWithCrossEntropyLoss extends anntensor.Layer {
  override val weightSize = 0
  override def outputSize(inputSize: Int): Int = inputSize
  override val inPlace = true
  override def model(weights: Tensor): anntensor.LayerModel =
    new SoftmaxLayerModelWithCrossEntropyLoss()
  override def initModel(weights: Tensor, random: Random): anntensor.LayerModel =
    new SoftmaxLayerModelWithCrossEntropyLoss()
}

private[anntensor] class SoftmaxLayerModelWithCrossEntropyLoss extends anntensor.LayerModel with LossFunction {

  private val epsilon = 1e-15
  private var epsilonMatrix: Tensor = null

  val weights: Tensor = DenseTensor(Array(0))

  def inplaceEval(x: BDM[Double], y: BDM[Double]): Unit = {
    var j = 0
    // find max value to make sure later that exponent is computable
    while (j < x.cols) {
      var i = 0
      var max = Double.MinValue
      while (i < x.rows) {
        if (x(i, j) > max) {
          max = x(i, j)
        }
        i += 1
      }
      var sum = 0.0
      i = 0
      while (i < x.rows) {
        val res = Math.exp(x(i, j) - max)
        y(i, j) = res
        sum += res
        i += 1
      }
      i = 0
      while (i < x.rows) {
        y(i, j) /= sum
        i += 1
      }
      j += 1
    }
  }

  override def eval(data: Tensor, output: Tensor): Unit = {
    // TODO: implement
    // inplaceEval(data, output)
  }
  override def prevDelta(nextDelta: Tensor, input: Tensor, delta: Tensor): Unit = {}

  override def grad(delta: Tensor, input: Tensor, cumGrad: Tensor): Unit = {}

  override def loss(output: Tensor, target: Tensor, delta: Tensor): Double = {
//    if (epsilonMatrix == null || epsilonMatrix.cols != target.cols) {
//      epsilonMatrix = BDM.fill[Double](target.rows, target.cols)(epsilon)
//    }
    //anntensor.UniversalFunction(output, target, delta, (o: Double, t: Double) => o - t)
    if (epsilonMatrix == null || epsilonMatrix.shape(1) != target.shape(1)) {
      epsilonMatrix = DenseTensor.fill(target.shape)(epsilon)
    }
    DenseTensor.applyFunction(output, target, delta, (o: Double, t: Double) => o - t)
    // TODO: Implement
    //-Bsum( target :* Blog(output + epsilonMatrix)) / output.cols
    0
  }
}

private[ml] class EmptyLayerWithSquaredError extends anntensor.Layer {
  override val weightSize = 0
  override def outputSize(inputSize: Int): Int = inputSize
  override val inPlace = true
  override def model(weights: Tensor): anntensor.LayerModel =
    new EmptyLayerModelWithSquaredError()
  override def initModel(weights: Tensor, random: Random): anntensor.LayerModel =
    new EmptyLayerModelWithSquaredError()
}

private[anntensor] class EmptyLayerModelWithSquaredError extends anntensor.LayerModel with LossFunction {

  val weights: Tensor = DenseTensor(Array(0))

  override def loss(output: Tensor, target: Tensor, delta: Tensor): Double = {
    DenseTensor.applyFunction(output, target, delta, (o: Double, t: Double) => o - t)
    // TODO: implement
    // Bsum(delta :* delta) / 2 / output.cols
    0
  }

  override def eval(data: Tensor, output: Tensor): Unit = {}
  override def prevDelta(nextDelta: Tensor, input: Tensor, delta: Tensor): Unit = {}
  override def grad(delta: Tensor, input: Tensor, cumGrad: Tensor): Unit = {}
}

private[ml] class SigmoidLayerWithCrossEntropyLoss extends anntensor.Layer {
  override val weightSize = 0
  override def outputSize(inputSize: Int): Int = inputSize
  override val inPlace = true
  override def model(weights: Tensor): anntensor.LayerModel =
    new SigmoidLayerModelWithCrossEntropyLoss()
  override def initModel(weights: Tensor, random: Random): anntensor.LayerModel =
    new SigmoidLayerModelWithCrossEntropyLoss()
}

private[anntensor] class SigmoidLayerModelWithCrossEntropyLoss
  extends anntensor.FunctionalLayerModel(new anntensor.FunctionalLayer(new anntensor.SigmoidFunction)) with LossFunction {
  // TODO: make a common place where ones matrices reside
  private var oneMatrix: Tensor = null
  private val epsilon = 1e-15
  private var epsilonMatrix: Tensor = null

  override def loss(output: Tensor, target: Tensor, delta: Tensor): Double = {
    if (oneMatrix == null || oneMatrix.shape(1) != target.shape(1)) {
      oneMatrix = DenseTensor.fill(target.shape)(1)
    }
    if (epsilonMatrix == null || epsilonMatrix.shape(1) != target.shape(1)) {
      epsilonMatrix = DenseTensor.fill(target.shape)(epsilon)
    }
    DenseTensor.applyFunction(output, target, delta, (o: Double, t: Double) => o - t)
    // TODO: implement
    0
    // NB: operation :* don't have execution priority over summation
    // TODO: is adding epsilon a good way to fight log(o) ?
//    -Bsum((target :* Blog(output + epsilonMatrix)) +
//      ((oneMatrix - target) :* Blog(oneMatrix - output + epsilonMatrix))) / output.cols
  }
}


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

package org.apache.spark.ml.feature

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.ann.{FeedForwardTrainer, EmptyLayerWithSquaredError, FeedForwardTopology}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.classification.MultilayerPerceptronParams
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors, Vector}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.types.{StructField, StructType}

import breeze.linalg.{DenseVector => BDV}


@Experimental
class StackedAutoencoder (override val uid: String)
  extends Estimator[StackedAutoencoderModel]
  with MultilayerPerceptronParams with AutoencoderParams {

  def this() = this(Identifiable.randomUID("stackedAutoencoder"))

  /** @group setParam */
  def setDataIn01Interval(value: Boolean): this.type = set(dataIn01Interval, value)

  // TODO: make sure that user understands how to set it. Make correctness check
  /** @group setParam */
  def setLayers(value: Array[Int]): this.type = set(layers, value)

  /** @group setParam */
  def setBlockSize(value: Int): this.type = set(blockSize, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-4.
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)

  /**
   * Set the seed for weights initialization.
   * @group setParam
   */
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Set the model weights.
   * @group setParam
   */
  def setWeights(value: Vector): this.type = set(weights, value)

  /**
   * Fits a model to the input data.
   */
  override def fit(dataset: DataFrame): StackedAutoencoderModel = {
    println("Topology:" + $(layers).mkString(" "))
    val stackedWeights = if ($(weights) == null) {
      FeedForwardTopology.multiLayerPerceptron($(layers)).model($(seed)).weights.toArray
    } else {
      $(weights).toArray
    }
    val data = dataset.select($(inputCol)).map { case Row(x: Vector) => (x, x) }
    val linearOutput = !$(dataIn01Interval)
    // Train autoencoder for each layer except the last
    var offset = 0
    for (i <- 0 until $(layers).length - 1) {
      val currentLayers = Array($(layers)(i), $(layers)(i + 1), $(layers)(i))
      println("Current:" + currentLayers.mkString(" "))
      val currentTopology = FeedForwardTopology.multiLayerPerceptron(currentLayers, false)
      val isLastWithLinear = i == $(layers).length - 3 && linearOutput
      if (isLastWithLinear) {
        currentTopology.layers(currentTopology.layers.length - 1) = new EmptyLayerWithSquaredError()
      }
      val FeedForwardTrainer = new FeedForwardTrainer(currentTopology, currentLayers(0), currentLayers.last)
      FeedForwardTrainer.LBFGSOptimizer
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
      FeedForwardTrainer.setStackSize($(blockSize))
      val currentModel = FeedForwardTrainer.train(data)
      val currentWeights = currentModel.weights.toArray
      var weightSize = currentTopology.layers(0).weightSize
      System.arraycopy(currentWeights, 0, stackedWeights, offset, weightSize)
      offset += weightSize
    }
    new StackedAutoencoderModel(uid + "model", $(layers), Vectors.dense(stackedWeights), linearOutput)
  }

  override def copy(extra: ParamMap): Estimator[StackedAutoencoderModel] = defaultCopy(extra)

  /**
   * :: DeveloperApi ::
   *
   * Derives the output schema from the input schema.
   */
  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }
}

@Experimental
class StackedAutoencoderModel private[ml] (
    override val uid: String,
    layers: Array[Int],
    weights: Vector,
    linearOutput: Boolean) extends Model[StackedAutoencoderModel] with AutoencoderParams {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  private val encoderModel = {
    val topology = FeedForwardTopology.multiLayerPerceptron(layers, false)
    topology.model(weights)
  }

  override def copy(extra: ParamMap): StackedAutoencoderModel = {
    copyValues(new StackedAutoencoderModel(uid, layers, weights, linearOutput), extra)
  }

  /**
   * Transforms the input dataset.
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val pcaOp = udf { encoderModel.predict _ }
    dataset.withColumn($(outputCol), pcaOp(col($(inputCol))))
  }

  /**
   * :: DeveloperApi ::
   *
   * Derives the output schema from the input schema.
   */
  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }
}

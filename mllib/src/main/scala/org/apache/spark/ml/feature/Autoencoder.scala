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
import org.apache.spark.ml.ann._
import org.apache.spark.ml.classification.MultilayerPerceptronParams
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.types.{StructField, StructType}

import breeze.linalg.{DenseVector => BDV}

/**
 * Params for [[Autoencoder]] and [[AutoencoderModel]].
 */
private[feature] trait AutoencoderParams extends Params with HasInputCol with HasOutputCol {
  /**
   * True if data is in [0, 1] interval.
   * Default: false
   * @group expertParam
   */
  final val dataIn01Interval: BooleanParam = new BooleanParam(this, "dataIn01Interval",
    "True if data is in [0, 1] interval." +
      " Sets the layer on the top of the autoencoder: linear + sigmoid (true) " +
      " or linear (false)")

  /** @group getParam */
  final def getDataIn01Interval: Boolean = $(dataIn01Interval)

}

/**
 * :: Experimental ::
 * Autoencoder.
 */
@Experimental
class Autoencoder (override val uid: String) extends Estimator[AutoencoderModel]
  with MultilayerPerceptronParams with AutoencoderParams  {

  def this() = this(Identifiable.randomUID("autoencoder"))

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
   * @param dataset dataset
   * @return encoder and decoder models
   */
  override def fit(dataset: DataFrame): AutoencoderModel = {
    val data = dataset.select($(inputCol)).map { case Row(x: Vector) => (x, x) }
    val linearOutput = !$(dataIn01Interval)
    val topology = FeedForwardTopology.multiLayerPerceptron($(layers), false)
    if (linearOutput) topology.layers(topology.layers.length - 1) = new EmptyLayerWithSquaredError()
    val FeedForwardTrainer = new FeedForwardTrainer(topology, $(layers)(0), $(layers).last)
    FeedForwardTrainer.LBFGSOptimizer
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
    FeedForwardTrainer.setStackSize($(blockSize))
    val encoderDecoderModel = FeedForwardTrainer.train(data)
    new AutoencoderModel(uid + "decoder", $(layers), encoderDecoderModel.weights, linearOutput)
  }

  override def copy(extra: ParamMap): Estimator[AutoencoderModel] = defaultCopy(extra)

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
/**
 * :: Experimental ::
 * Autoencoder model.
 *
 * @param layers array of layer sizes including input and output
 * @param weights weights (or parameters) of the model
 */
@Experimental
class AutoencoderModel private[ml] (
    override val uid: String,
    layers: Array[Int],
    weights: Vector,
    linearOutput: Boolean) extends Model[AutoencoderModel] with AutoencoderParams {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  private var coderWeightSize = 0
  private val decoderModel = {
    val topology = FeedForwardTopology.multiLayerPerceptron(layers.tail, false)
    for (layer <- topology.layers) {
      coderWeightSize += layer.weightSize
    }
    if (linearOutput) topology.layers(topology.layers.length - 1) = new EmptyLayerWithSquaredError()
    val decoderWeights =
      Vectors.fromBreeze(new BDV(weights.toArray, weights.size - coderWeightSize))
    topology.model(decoderWeights)
  }
  private val encoderModel = {
    val topology = FeedForwardTopology.multiLayerPerceptron(layers.init, false)
    val encoderWeights =
      Vectors.fromBreeze(new BDV(weights.toArray, 0, 1, coderWeightSize))
    topology.model(encoderWeights)
  }


  override def copy(extra: ParamMap): AutoencoderModel = {
    copyValues(new AutoencoderModel(uid, layers, weights, linearOutput), extra)
  }

  /**
   * Transforms the input dataset.
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val pcaOp = udf { encoderModel.predict _ }
    dataset.withColumn($(outputCol), pcaOp(col($(inputCol))))
  }

  def encode(dataset: DataFrame): DataFrame = transform(dataset)

  def decode(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val pcaOp = udf { decoderModel.predict _ }
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

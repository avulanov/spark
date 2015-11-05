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

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors, Vector}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.Row

import breeze.linalg.{DenseMatrix => BDM}

class AutoencoderSuite  extends SparkFunSuite with MLlibTestSparkContext {

  // using data from https://inst.eecs.berkeley.edu/~cs182/sp08/assignments/a3-tlearn.html
  val binaryData = Seq(
    Vectors.dense(Array(1.0, 0.0, 0.0, 0.0)),
    Vectors.dense(Array(0.0, 1.0, 0.0, 0.0)),
    Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
    Vectors.dense(Array(0.0, 0.0, 0.0, 1.0)))

  val real01Data = Seq(
    Vectors.dense(Array(0.5, 0.5, 0.5, 0.6)),
    Vectors.dense(Array(0.5, 0.5, 0.5, 0.55)),
    Vectors.dense(Array(0.5, 0.5, 0.5, 0.45)),
    Vectors.dense(Array(0.9, 0.9, 0.9, 0.9)))

  val realData = Seq(
    Vectors.dense(Array(10.0, 0.0, 0.0, 0.0)),
    Vectors.dense(Array(0.0, 1.0, 0.0, 0.0)),
    Vectors.dense(Array(0.0, 0.0, 10.0, 0.0)),
    Vectors.dense(Array(0.0, 0.0, 0.0, 10.0)))

  test("Autoencoder suite for binary input") {
    // TODO: implement autoencoder test for real in [0;1) and (-inf;+inf)
    val rdd = sc.parallelize(realData, 1).map(x => Tuple1(x))
    val df = sqlContext.createDataFrame(rdd).toDF("input")
    val autoencoder = new Autoencoder()
      .setLayers(Array(4, 3, 4, 3))
      .setBlockSize(1)
//      .setMaxIter(1000)
//      .setOptimizer("GD")
//      .setLearningRate(1.0)
      .setOptimizer("LBFGS")
      .setMaxIter(202)
      .setSeed(111L)
      .setTol(1e-15)
      .setInputCol("input")
      .setOutputCol("output")
    // TODO: find a way to inherit the input and output parameter value from estimator
    val (encoderModel, decoderModel) = autoencoder.fit2(df)
    encoderModel.setInputCol("input").setOutputCol("encoded")
    decoderModel.setInputCol("encoded").setOutputCol("decoded")
    // TODO: how the check that output makes sense?
    val encodedData = encoderModel.transform(df)
    val decodedData = decoderModel.transform(encodedData)
    val result = decodedData.map {
      case Row(input: Vector, encoded: Vector, decoded: Vector) =>
        Vectors.dense(decoded.toArray.map(x => math.round(x * 100).toDouble / 100))
    }.collect()
    println("decoded rounded")
    result.foreach(println)
    val eps = 0.01
    println("Original, encoded & decoded")
    decodedData.collect.foreach(println)
  }
}

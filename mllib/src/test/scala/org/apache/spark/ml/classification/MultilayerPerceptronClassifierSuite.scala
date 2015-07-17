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

package org.apache.spark.ml.classification

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.Row

class MultilayerPerceptronClassifierSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("XOR") {
    val inputs = Array[Array[Double]](
      Array[Double](0, 0),
      Array[Double](0, 1),
      Array[Double](1, 0),
      Array[Double](1, 1)
    )
    val outputs = Array[Double](0, 1, 1, 0)
    val data = inputs.zip(outputs).map{ case(input, output) =>
      new LabeledPoint(output, Vectors.dense(input))}
    val rddData = sc.parallelize(data, 2)
    val layers = Array[Int](2, 5, 2)
    val dataFrame = sqlContext.createDataFrame(rddData)
    val trainer = new MultilayerPerceptronClassifier("mlpc")
      .setLayers(layers.map(_.toDouble))
      .setBlockSize(1)
      .setSeed(11L)
      .setMaxIter(100)
    val model = trainer.fit(dataFrame)
    val result = model.transform(dataFrame)
    val predictionAndLabels = result.select("prediction", "label").collect()
    assert(predictionAndLabels.forall { case Row (p: Double, l: Double) =>
      (p - l) == 0 })
  }
}

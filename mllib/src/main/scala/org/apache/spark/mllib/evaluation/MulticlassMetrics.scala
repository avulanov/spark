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

package org.apache.spark.mllib.evaluation

import scala.collection.Map

import org.apache.spark.SparkContext._
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.apache.spark.rdd.RDD

/**
 * ::Experimental::
 * Evaluator for multiclass classification.
 *
 * @param predictionAndLabels an RDD of (prediction, label) pairs.
 */
@Experimental
class MulticlassMetrics(predictionAndLabels: RDD[(Double, Double)]) {

  private lazy val labelCountByClass: Map[Double, Long] = predictionAndLabels.values.countByValue()
  private lazy val labelCount: Long = labelCountByClass.values.sum
  private lazy val tpByClass: Map[Double, Int] = predictionAndLabels
    .map { case (prediction, label) =>
      (label, if (label == prediction) 1 else 0)
    }.reduceByKey(_ + _)
    .collectAsMap()
  private lazy val fpByClass: Map[Double, Int] = predictionAndLabels
    .map { case (prediction, label) =>
      (prediction, if (prediction != label) 1 else 0)
    }.reduceByKey(_ + _)
    .collectAsMap()
  private lazy val confusions = predictionAndLabels
    .map { case (prediction, label) =>
      ((prediction, label), 1)
  }.reduceByKey(_ + _).collectAsMap()

  /**
   * Returns confusion matrix:
   * predicted classes are in columns,
   * they are ordered by class label ascending,
   * as in "labels"
   */
  def confusionMatrix: Matrix = {
    val transposedFlatMatrix = Array.ofDim[Double](labels.size * labels.size)
    val n = labels.size
    var i, j = 0
    while(i < n){
      j = 0
      while(j < n){
        transposedFlatMatrix(i * labels.size + j)
          = confusions.getOrElse((labels(i), labels(j)), 0).toDouble
        j += 1
      }
      i += 1
    }
    Matrices.dense(labels.size, labels.size, transposedFlatMatrix)
  }

  /**
   * Returns true positive rate for a given label (category)
   * @param label the label.
   */
  def truePositiveRate(label: Double): Double = recall(label)

  /**
   * Returns false positive rate for a given label (category)
   * @param label the label.
   */
  def falsePositiveRate(label: Double): Double = {
    val fp = fpByClass.getOrElse(label, 0)
    fp.toDouble / (labelCount - labelCountByClass(label))
  }

  /**
   * Returns precision for a given label (category)
   * @param label the label.
   */
  def precision(label: Double): Double = {
    val tp = tpByClass(label)
    val fp = fpByClass.getOrElse(label, 0)
    if (tp + fp == 0) 0 else tp.toDouble / (tp + fp)
  }

  /**
   * Returns recall for a given label (category)
   * @param label the label.
   */
  def recall(label: Double): Double = tpByClass(label).toDouble / labelCountByClass(label)

  /**
   * Returns f-measure for a given label (category)
   * @param label the label.
   * @param beta the beta parameter.
   */
  def fMeasure(label: Double, beta: Double): Double = {
    val p = precision(label)
    val r = recall(label)
    val betaSqrd = beta * beta
    if (p + r == 0) 0 else (1 + betaSqrd) * p * r / (betaSqrd * p + r)
  }

  /**
   * Returns f1-measure for a given label (category)
   * @param label the label.
   */
  def fMeasure(label: Double): Double = fMeasure(label, 1.0)

  /**
   * Returns precision
   */
  lazy val precision: Double = tpByClass.values.sum.toDouble / labelCount

  /**
   * Returns recall
   * (equals to precision for multiclass classifier
   * because sum of all false positives is equal to sum
   * of all false negatives)
   */
  lazy val recall: Double = precision

  /**
   * Returns f-measure
   * (equals to precision and recall because precision equals recall)
   */
  lazy val fMeasure: Double = precision

  /**
   * Returns weighted true positive rate
   * (equals to precision, recall and f-measure)
   */
  lazy val weightedTruePositiveRate: Double = weightedRecall

  /**
   * Returns weighted false positive rate
   */
  lazy val weightedFalsePositiveRate: Double = labelCountByClass.map { case (category, count) =>
    falsePositiveRate(category) * count.toDouble / labelCount
  }.sum

  /**
   * Returns weighted averaged recall
   * (equals to precision, recall and f-measure)
   */
  lazy val weightedRecall: Double = labelCountByClass.map { case (category, count) =>
    recall(category) * count.toDouble / labelCount
  }.sum

  /**
   * Returns weighted averaged precision
   */
  lazy val weightedPrecision: Double = labelCountByClass.map { case (category, count) =>
    precision(category) * count.toDouble / labelCount
  }.sum

  /**
   * Returns weighted averaged f-measure
   * @param beta the beta parameter.
   */
  def weightedFMeasure(beta: Double): Double = labelCountByClass.map { case (category, count) =>
    fMeasure(category, beta) * count.toDouble / labelCount
  }.sum

  /**
   * Returns weighted averaged f1-measure
   */
  lazy val weightedFMeasure: Double = labelCountByClass.map { case (category, count) =>
    fMeasure(category, 1.0) * count.toDouble / labelCount
  }.sum

  /**
   * Returns the sequence of labels in ascending order
   */
  lazy val labels: Array[Double] = tpByClass.keys.toArray.sorted
}

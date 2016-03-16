package org.apache.spark.ml.anntensor

import breeze.linalg.{sum, DenseMatrix}
import org.apache.spark.ml.ann.DenseTensor
import org.apache.spark.ml.anntensor.Math.NumberLike.NumberLikeDouble
import org.scalatest.FunSuite

import scala.util.Random

/**
  * Created by ulanov on 3/14/2016.
  */
class DenseTensorSpeedSuite extends FunSuite {

  test ("ops") {
    val max = 1025
    val min = 1024
    var i = min
    var tBreeze = 0L
    while (i < max) {
      println(i)
      val a = DenseMatrix.fill[Double](i, i)(Random.nextDouble())
      val b = DenseMatrix.fill[Double](i, i)(Random.nextDouble())
      val t1 = System.nanoTime()
      val c = sum(a)//a :* b
      tBreeze = tBreeze + (System.nanoTime() - t1)
      i = i * 2
    }
    println("Breeze: " + tBreeze / 1e9 + " s.")
    var k = min
    var tArray = 0L
    val m = new My[Double]()
    while (k < max) {
      println(k)
      val a = Array.fill[Double](k * k)(Random.nextDouble())
      val b = Array.fill[Double](k * k)(Random.nextDouble())
      val t = System.nanoTime()
      val c = new Array[Double](k * k)
      var p = 0
      while (p < k * k) {
        c(p) = m.plus(a(p), b(p))
        p += 1
      }
      tArray = tArray + (System.nanoTime() - t)
      k = k * 2
    }
    println("Array: " + tBreeze / 1e9 + " s.")

    var j = min
    var tTensor = 0L
    while (j < max) {
      println(j)
      val x = DenseTensor.fill[Double](Array(j, j))(Random.nextDouble())
      val y = DenseTensor.fill[Double](Array(j, j))(Random.nextDouble())
      val t2 = System.nanoTime()
      val z = x.sum//x :* y
      tTensor = tTensor + (System.nanoTime() - t2)
      j = j * 2
    }
    println("DT: " + tTensor / 1e9 + " s.")
  }
}

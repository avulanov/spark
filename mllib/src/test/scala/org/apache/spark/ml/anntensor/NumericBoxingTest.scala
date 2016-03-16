package org.apache.spark.ml.anntensor

class NumericBoxingTest[@specialized(Double, Float) T : Numeric] {
  lazy val numOps = implicitly[Numeric[T]]
  def plus(x: T, y: T): T = numOps.plus(x, y)
}
